"""
Meta-Learning Hyperparameter Optimization for High Leverage Trading

This module provides a specialized meta-learning HPO system designed specifically
for high leverage trading scenarios with market regime awareness, risk management,
and trading-specific optimization strategies.

Key Features:
- Trading-specific meta-feature extraction
- Market regime-aware optimization
- Risk-aware hyperparameter search
- High leverage specific constraints
- Regime transition handling
- Financial performance optimization
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import logging
import json
import sqlite3
from pathlib import Path
import warnings

from .hpo_utils import HyperparameterOptimization
from .trading_meta_features import TradingMetaFeaturesExtractor
from ..math_validation import safe_divide, safe_log
from ..common_operations import create_fallback_logger

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available - limited HPO functionality")


class TradingOptimizationHistoryDB:
    """Database for storing trading optimization history."""
    
    def __init__(self, db_path: str = "trading_hpo_history.db"):
        """Initialize trading optimization history database."""
        self.db_path = db_path
        self.logger = logger.getChild('TradingHistoryDB')
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the database schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create optimization results table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS optimization_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        dataset_meta_features TEXT NOT NULL,
                        market_regime TEXT,
                        search_space TEXT NOT NULL,
                        best_params TEXT NOT NULL,
                        best_score REAL NOT NULL,
                        optimization_time REAL,
                        risk_metrics TEXT,
                        regime_transition_detected BOOLEAN,
                        leverage_used REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create parameter importance table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS parameter_importance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        optimization_id INTEGER,
                        parameter_name TEXT NOT NULL,
                        importance_score REAL NOT NULL,
                        regime_specific BOOLEAN DEFAULT FALSE,
                        FOREIGN KEY (optimization_id) REFERENCES optimization_results (id)
                    )
                ''')
                
                # Create regime patterns table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS regime_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        regime_type TEXT NOT NULL,
                        meta_features TEXT NOT NULL,
                        optimal_params TEXT NOT NULL,
                        performance_score REAL NOT NULL,
                        confidence_score REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                self.logger.info("✅ Trading optimization history database initialized")
                
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")
    
    def store_optimization_result(self, 
                                dataset_meta_features: Dict[str, float],
                                model_type: str,
                                market_regime: str,
                                search_space: Dict[str, Any],
                                best_params: Dict[str, Any],
                                best_score: float,
                                optimization_time: float,
                                risk_metrics: Optional[Dict[str, float]] = None,
                                regime_transition_detected: bool = False,
                                leverage_used: float = 1.0):
        """Store optimization result in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO optimization_results 
                    (timestamp, model_type, dataset_meta_features, market_regime, 
                     search_space, best_params, best_score, optimization_time, 
                     risk_metrics, regime_transition_detected, leverage_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    model_type,
                    json.dumps(dataset_meta_features),
                    market_regime,
                    json.dumps(search_space),
                    json.dumps(best_params),
                    best_score,
                    optimization_time,
                    json.dumps(risk_metrics) if risk_metrics else None,
                    regime_transition_detected,
                    leverage_used
                ))
                
                optimization_id = cursor.lastrowid
                conn.commit()
                
                self.logger.info(f"✅ Stored optimization result with ID: {optimization_id}")
                return optimization_id
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store optimization result: {e}")
            return None
    
    def find_similar_trading_datasets(self, 
                                    target_meta_features: Dict[str, float],
                                    model_type: str,
                                    market_regime: Optional[str] = None,
                                    similarity_threshold: float = 0.7) -> List[Dict]:
        """Find similar trading datasets based on meta-features."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query
                query = '''
                    SELECT id, dataset_meta_features, market_regime, best_params, 
                           best_score, risk_metrics, leverage_used
                    FROM optimization_results 
                    WHERE model_type = ?
                '''
                params = [model_type]
                
                if market_regime:
                    query += ' AND market_regime = ?'
                    params.append(market_regime)
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                # Calculate similarity
                similar_results = []
                for result in results:
                    result_id, meta_features_json, regime, best_params_json, score, risk_metrics_json, leverage = result
                    
                    meta_features = json.loads(meta_features_json)
                    similarity = self._calculate_trading_similarity(target_meta_features, meta_features)
                    
                    if similarity >= similarity_threshold:
                        similar_results.append({
                            'id': result_id,
                            'similarity': similarity,
                            'market_regime': regime,
                            'best_params': json.loads(best_params_json),
                            'best_score': score,
                            'risk_metrics': json.loads(risk_metrics_json) if risk_metrics_json else {},
                            'leverage_used': leverage
                        })
                
                # Sort by similarity
                similar_results.sort(key=lambda x: x['similarity'], reverse=True)
                return similar_results
                
        except Exception as e:
            self.logger.error(f"❌ Failed to find similar datasets: {e}")
            return []
    
    def get_regime_specific_parameters(self, 
                                     market_regime: str,
                                     model_type: str,
                                     min_confidence: float = 0.6) -> Dict[str, Any]:
        """Get regime-specific optimal parameters."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT optimal_params, performance_score, confidence_score
                    FROM regime_patterns
                    WHERE regime_type = ? AND performance_score >= ?
                    ORDER BY confidence_score DESC
                ''', (market_regime, min_confidence))
                
                results = cursor.fetchall()
                if not results:
                    return {}
                
                # Weight parameters by confidence
                weighted_params = {}
                total_weight = 0
                
                for optimal_params_json, performance_score, confidence_score in results:
                    optimal_params = json.loads(optimal_params_json)
                    weight = confidence_score * performance_score
                    total_weight += weight
                    
                    for param_name, param_value in optimal_params.items():
                        if param_name not in weighted_params:
                            weighted_params[param_name] = 0
                        weighted_params[param_name] += param_value * weight
                
                # Normalize by total weight
                if total_weight > 0:
                    for param_name in weighted_params:
                        weighted_params[param_name] /= total_weight
                
                return weighted_params
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get regime-specific parameters: {e}")
            return {}
    
    def _calculate_trading_similarity(self, 
                                    features1: Dict[str, float], 
                                    features2: Dict[str, float]) -> float:
        """Calculate similarity between trading datasets."""
        try:
            # Key trading features for similarity
            key_features = [
                'n_observations', 'returns_std', 'returns_skewness', 'returns_kurtosis',
                'sharpe_ratio', 'max_drawdown', 'volatility_clustering', 'trend_strength',
                'leverage_risk', 'market_correlation', 'liquidity_risk'
            ]
            
            similarities = []
            for feature in key_features:
                if feature in features1 and feature in features2:
                    val1, val2 = features1[feature], features2[feature]
                    
                    # Handle different scales
                    if feature in ['n_observations']:
                        # Log scale for large numbers
                        val1, val2 = safe_log(val1), safe_log(val2)
                    
                    # Calculate similarity (1 - normalized difference)
                    max_val = max(abs(val1), abs(val2))
                    if max_val > 0:
                        similarity = 1.0 - abs(val1 - val2) / max_val
                        similarities.append(max(0, similarity))
            
            return float(np.mean(similarities)) if similarities else 0.0
            
        except Exception as e:
            self.logger.warning(f"Similarity calculation failed: {e}")
            return 0.0


class MetaLearningTradingHPO(HyperparameterOptimization):
    """Meta-learning hyperparameter optimization for high leverage trading."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize meta-learning trading HPO."""
        super().__init__(config)
        self.logger = logger.getChild('MetaLearningTradingHPO')
        
        # Trading-specific configuration
        self.trading_config = self.config.get('trading', {})
        self.max_leverage = self.trading_config.get('max_leverage', 10.0)
        self.risk_tolerance = self.trading_config.get('risk_tolerance', 0.05)
        self.regime_awareness = self.trading_config.get('regime_awareness', True)
        self.leverage_aware_optimization = self.trading_config.get('leverage_aware_optimization', True)
        
        # Initialize trading-specific components
        self.meta_features_extractor = TradingMetaFeaturesExtractor(self.trading_config)
        self.history_db = TradingOptimizationHistoryDB(
            self.trading_config.get('history_db_path', 'trading_hpo_history.db')
        )
        
        # Trading-specific search spaces
        self.trading_search_spaces = self._initialize_trading_search_spaces()
        
        # Risk constraints
        self.risk_constraints = self._initialize_risk_constraints()
    
    def _initialize_trading_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Initialize trading-specific search spaces."""
        return {
            'xgboost_trading': {
                'max_depth': {'type': 'int', 'low': 3, 'high': 12},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'gamma': {'type': 'float', 'low': 0, 'high': 10},
                'reg_alpha': {'type': 'float', 'low': 0, 'high': 20},
                'reg_lambda': {'type': 'float', 'low': 0, 'high': 20},
                'min_child_weight': {'type': 'int', 'low': 1, 'high': 20}
            },
            'lightgbm_trading': {
                'num_leaves': {'type': 'int', 'low': 10, 'high': 200},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                'feature_fraction': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'bagging_fraction': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'bagging_freq': {'type': 'int', 'low': 1, 'high': 10},
                'min_child_samples': {'type': 'int', 'low': 5, 'high': 100},
                'lambda_l1': {'type': 'float', 'low': 0, 'high': 20},
                'lambda_l2': {'type': 'float', 'low': 0, 'high': 20},
                'min_data_in_leaf': {'type': 'int', 'low': 5, 'high': 50}
            },
            'neural_network_trading': {
                'hidden_layers': {'type': 'int', 'low': 1, 'high': 5},
                'hidden_units': {'type': 'int', 'low': 32, 'high': 512},
                'learning_rate': {'type': 'float', 'low': 0.0001, 'high': 0.01},
                'dropout_rate': {'type': 'float', 'low': 0.0, 'high': 0.5},
                'batch_size': {'type': 'int', 'low': 16, 'high': 256},
                'epochs': {'type': 'int', 'low': 10, 'high': 200},
                'l1_regularization': {'type': 'float', 'low': 0.0, 'high': 0.1},
                'l2_regularization': {'type': 'float', 'low': 0.0, 'high': 0.1}
            }
        }
    
    def _initialize_risk_constraints(self) -> Dict[str, Any]:
        """Initialize risk constraints for high leverage trading."""
        return {
            'max_drawdown_threshold': 0.15,  # 15% max drawdown
            'min_sharpe_ratio': 1.0,  # Minimum Sharpe ratio
            'max_var_95': -0.05,  # Maximum 5% daily VaR
            'max_leverage_risk': 0.8,  # Maximum leverage risk score
            'min_calmar_ratio': 0.5,  # Minimum Calmar ratio
            'max_correlation_with_market': 0.8  # Maximum market correlation
        }
    
    def trading_meta_learning_optimization(self, 
                                         model_factory: Callable,
                                         price_data: pd.DataFrame,
                                         target_data: pd.Series,
                                         model_type: str,
                                         market_regime: Optional[str] = None,
                                         leverage_factor: float = 1.0,
                                         n_trials: int = 100) -> Dict[str, Any]:
        """
        Perform meta-learning optimization for trading models.
        
        Args:
            model_factory: Function that creates model with given parameters
            price_data: OHLCV price data
            target_data: Target variable (returns, signals, etc.)
            model_type: Type of model ('xgboost_trading', 'lightgbm_trading', etc.)
            market_regime: Current market regime
            leverage_factor: Leverage factor being used
            n_trials: Number of optimization trials
            
        Returns:
            Meta-learning optimization results
        """
        try:
            self.logger.info(f"🏦 Starting trading meta-learning optimization for {model_type}")
            
            # 1. Extract trading meta-features
            dataset_meta_features = self.meta_features_extractor.extract_trading_meta_features(
                price_data=price_data,
                returns_data=target_data if 'return' in str(type(target_data)).lower() else None
            )
            
            # 2. Detect market regime if not provided
            if market_regime is None and self.regime_awareness:
                market_regime = self._detect_current_market_regime(dataset_meta_features)
            
            # 3. Generate meta-learning search space
            search_space = self._generate_trading_meta_learning_search_space(
                model_type=model_type,
                dataset_meta_features=dataset_meta_features,
                market_regime=market_regime,
                leverage_factor=leverage_factor,
                optimization_budget=n_trials
            )
            
            # 4. Perform risk-aware optimization
            optimization_results = self._risk_aware_trading_optimization(
                model_factory=model_factory,
                price_data=price_data,
                target_data=target_data,
                search_space=search_space,
                dataset_meta_features=dataset_meta_features,
                market_regime=market_regime,
                leverage_factor=leverage_factor,
                n_trials=n_trials
            )
            
            # 5. Store results for future meta-learning
            risk_metrics = self._calculate_trading_risk_metrics(
                optimization_results, dataset_meta_features, leverage_factor
            )
            
            self.history_db.store_optimization_result(
                dataset_meta_features=dataset_meta_features,
                model_type=model_type,
                market_regime=market_regime or 'unknown',
                search_space=search_space,
                best_params=optimization_results['best_params'],
                best_score=optimization_results['best_score'],
                optimization_time=optimization_results.get('optimization_time', 0),
                risk_metrics=risk_metrics,
                regime_transition_detected=self._detect_regime_transition(dataset_meta_features),
                leverage_used=leverage_factor
            )
            
            # 6. Add trading-specific insights
            optimization_results['trading_insights'] = {
                'market_regime': market_regime,
                'leverage_factor': leverage_factor,
                'risk_metrics': risk_metrics,
                'meta_learning_confidence': self._calculate_meta_learning_confidence(
                    dataset_meta_features, model_type, market_regime
                ),
                'regime_transition_detected': self._detect_regime_transition(dataset_meta_features),
                'similar_datasets_found': len(self.history_db.find_similar_trading_datasets(
                    dataset_meta_features, model_type, market_regime
                ))
            }
            
            self.logger.info(f"✅ Trading meta-learning optimization completed - "
                           f"Best score: {optimization_results['best_score']:.4f}, "
                           f"Regime: {market_regime}, Leverage: {leverage_factor}x")
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"❌ Trading meta-learning optimization failed: {e}")
            # Fallback to standard optimization
            return self.bayesian_optimization(
                model_factory, 
                price_data.values, 
                target_data.values, 
                n_trials=n_trials
            )
    
    def _generate_trading_meta_learning_search_space(self, 
                                                   model_type: str,
                                                   dataset_meta_features: Dict[str, float],
                                                   market_regime: Optional[str],
                                                   leverage_factor: float,
                                                   optimization_budget: int) -> Dict[str, Any]:
        """Generate meta-learning search space for trading."""
        try:
            self.logger.info("🧠 Generating trading meta-learning search space")
            
            # 1. Find similar trading datasets
            similar_optimizations = self.history_db.find_similar_trading_datasets(
                dataset_meta_features, model_type, market_regime
            )
            
            # 2. Get regime-specific parameters
            regime_params = {}
            if market_regime:
                regime_params = self.history_db.get_regime_specific_parameters(
                    market_regime, model_type
                )
            
            # 3. Start with base search space
            if model_type in self.trading_search_spaces:
                base_search_space = self.trading_search_spaces[model_type].copy()
            else:
                base_search_space = self.default_search_spaces.get(model_type.replace('_trading', ''), {})
            
            # 4. Adapt based on similar optimizations
            if similar_optimizations:
                adapted_search_space = self._adapt_search_space_from_history(
                    base_search_space, similar_optimizations, dataset_meta_features
                )
            else:
                adapted_search_space = base_search_space
            
            # 5. Apply regime-specific adaptations
            if regime_params:
                adapted_search_space = self._apply_regime_specific_adaptations(
                    adapted_search_space, regime_params
                )
            
            # 6. Apply leverage-specific constraints
            if self.leverage_aware_optimization:
                adapted_search_space = self._apply_leverage_constraints(
                    adapted_search_space, leverage_factor, dataset_meta_features
                )
            
            # 7. Apply risk-aware adaptations
            adapted_search_space = self._apply_risk_aware_adaptations(
                adapted_search_space, dataset_meta_features
            )
            
            # 8. Optimize for budget
            adapted_search_space = self._optimize_search_space_for_budget(
                adapted_search_space, optimization_budget
            )
            
            self.logger.info(f"✅ Generated trading meta-learning search space with {len(adapted_search_space)} parameters")
            return adapted_search_space
            
        except Exception as e:
            self.logger.error(f"❌ Trading meta-learning search space generation failed: {e}")
            return self.trading_search_spaces.get(model_type, {})
    
    def _risk_aware_trading_optimization(self, 
                                       model_factory: Callable,
                                       price_data: pd.DataFrame,
                                       target_data: pd.Series,
                                       search_space: Dict[str, Any],
                                       dataset_meta_features: Dict[str, float],
                                       market_regime: Optional[str],
                                       leverage_factor: float,
                                       n_trials: int) -> Dict[str, Any]:
        """Perform risk-aware optimization for trading."""
        try:
            self.logger.info("⚠️ Starting risk-aware trading optimization")
            
            if not OPTUNA_AVAILABLE:
                raise ImportError("Optuna required for risk-aware optimization")
            
            def risk_aware_objective(trial):
                # Sample parameters
                params = {}
                for param_name, param_config in search_space.items():
                    if isinstance(param_config, dict):
                        param_type = param_config.get('type', 'float')
                        if param_type == 'float':
                            params[param_name] = trial.suggest_float(
                                param_name,
                                param_config['low'],
                                param_config['high']
                            )
                        elif param_type == 'int':
                            params[param_name] = trial.suggest_int(
                                param_name,
                                param_config['low'],
                                param_config['high']
                            )
                        elif param_type == 'categorical':
                            params[param_name] = trial.suggest_categorical(
                                param_name,
                                param_config['choices']
                            )
                
                # Create and train model
                model = model_factory(**params)
                
                # Evaluate with risk constraints
                score = self._evaluate_trading_model_with_risk(
                    model, price_data, target_data, dataset_meta_features, leverage_factor
                )
                
                # Report intermediate results
                trial.report(score, step=trial.number)
                
                # Prune if risk constraints violated
                if self._check_risk_constraints_violated(score, dataset_meta_features, leverage_factor):
                    raise optuna.TrialPruned()
                
                return score
            
            # Create study with risk-aware pruner
            sampler = TPESampler()
            pruner = MedianPruner()
            
            study = optuna.create_study(
                direction='maximize',
                sampler=sampler,
                pruner=pruner
            )
            
            study.optimize(risk_aware_objective, n_trials=n_trials)
            
            results = {
                'best_params': study.best_params,
                'best_score': study.best_value,
                'n_trials': len(study.trials),
                'optimization_curve': [t.value for t in study.trials if t.value is not None],
                'risk_constraints_satisfied': True
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Risk-aware trading optimization failed: {e}")
            return {'error': str(e)}
    
    def _evaluate_trading_model_with_risk(self, 
                                        model: Any,
                                        price_data: pd.DataFrame,
                                        target_data: pd.Series,
                                        dataset_meta_features: Dict[str, float],
                                        leverage_factor: float) -> float:
        """Evaluate trading model with risk considerations."""
        try:
            # Prepare data for evaluation
            X = price_data.values if hasattr(price_data, 'values') else price_data
            y = target_data.values if hasattr(target_data, 'values') else target_data
            
            # Train model
            model.fit(X, y)
            
            # Make predictions
            predictions = model.predict(X)
            
            # Calculate base performance score
            base_score = self._calculate_trading_performance_score(y, predictions)
            
            # Apply risk adjustments
            risk_adjusted_score = self._apply_risk_adjustments(
                base_score, y, predictions, dataset_meta_features, leverage_factor
            )
            
            return risk_adjusted_score
            
        except Exception as e:
            self.logger.warning(f"Trading model evaluation failed: {e}")
            return 0.0
    
    def _calculate_trading_performance_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate trading-specific performance score."""
        try:
            # For trading, we care about:
            # 1. Direction accuracy (for classification)
            # 2. Return prediction accuracy (for regression)
            # 3. Risk-adjusted returns
            
            if len(np.unique(y_true)) <= 10:  # Classification
                # Direction accuracy
                direction_accuracy = np.mean((y_true > 0) == (y_pred > 0))
                return direction_accuracy
            else:  # Regression
                # Correlation with actual returns
                correlation = np.corrcoef(y_true, y_pred)[0, 1]
                return correlation if not np.isnan(correlation) else 0.0
                
        except Exception as e:
            self.logger.warning(f"Trading performance score calculation failed: {e}")
            return 0.0
    
    def _apply_risk_adjustments(self, 
                              base_score: float,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              dataset_meta_features: Dict[str, float],
                              leverage_factor: float) -> float:
        """Apply risk-based adjustments to performance score."""
        try:
            adjusted_score = base_score
            
            # 1. Drawdown penalty
            max_drawdown = dataset_meta_features.get('max_drawdown', 0)
            if max_drawdown < -self.risk_constraints['max_drawdown_threshold']:
                adjusted_score *= 0.5  # Heavy penalty for high drawdown
            
            # 2. Leverage risk penalty
            leverage_risk = dataset_meta_features.get('leverage_risk', 0)
            if leverage_risk > self.risk_constraints['max_leverage_risk']:
                adjusted_score *= 0.7  # Penalty for high leverage risk
            
            # 3. Sharpe ratio bonus
            sharpe_ratio = dataset_meta_features.get('sharpe_ratio', 0)
            if sharpe_ratio > self.risk_constraints['min_sharpe_ratio']:
                adjusted_score *= 1.1  # Bonus for good Sharpe ratio
            
            # 4. Market correlation penalty (for diversification)
            market_correlation = abs(dataset_meta_features.get('market_correlation', 0))
            if market_correlation > self.risk_constraints['max_correlation_with_market']:
                adjusted_score *= 0.8  # Penalty for high market correlation
            
            return max(0.0, adjusted_score)
            
        except Exception as e:
            self.logger.warning(f"Risk adjustment failed: {e}")
            return base_score
    
    def _check_risk_constraints_violated(self, 
                                       score: float,
                                       dataset_meta_features: Dict[str, float],
                                       leverage_factor: float) -> bool:
        """Check if risk constraints are violated."""
        try:
            # Check various risk constraints
            max_drawdown = dataset_meta_features.get('max_drawdown', 0)
            if max_drawdown < -self.risk_constraints['max_drawdown_threshold']:
                return True
            
            leverage_risk = dataset_meta_features.get('leverage_risk', 0)
            if leverage_risk > self.risk_constraints['max_leverage_risk']:
                return True
            
            var_95 = dataset_meta_features.get('var_95', 0)
            if var_95 < self.risk_constraints['max_var_95']:
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Risk constraint check failed: {e}")
            return False
    
    def _detect_current_market_regime(self, dataset_meta_features: Dict[str, float]) -> str:
        """Detect current market regime from meta-features."""
        try:
            volatility = dataset_meta_features.get('returns_std', 0)
            trend_strength = dataset_meta_features.get('trend_strength', 0)
            skewness = dataset_meta_features.get('returns_skewness', 0)
            
            # Simple regime detection
            if volatility > 0.02:  # High volatility
                if trend_strength > 0:
                    return 'high_vol_bull'
                else:
                    return 'high_vol_bear'
            else:  # Low volatility
                if trend_strength > 0:
                    return 'low_vol_bull'
                else:
                    return 'low_vol_bear'
                    
        except Exception as e:
            self.logger.warning(f"Market regime detection failed: {e}")
            return 'unknown'
    
    def _detect_regime_transition(self, dataset_meta_features: Dict[str, float]) -> bool:
        """Detect if regime transition is occurring."""
        try:
            # Look for signs of regime transition
            volatility_clustering = dataset_meta_features.get('volatility_clustering', 0)
            regime_transition_frequency = dataset_meta_features.get('regime_transition_frequency', 0)
            
            # High volatility clustering and transition frequency indicate regime change
            return (volatility_clustering > 0.3 and regime_transition_frequency > 0.1)
            
        except Exception as e:
            self.logger.warning(f"Regime transition detection failed: {e}")
            return False
    
    def _calculate_trading_risk_metrics(self, 
                                      optimization_results: Dict[str, Any],
                                      dataset_meta_features: Dict[str, float],
                                      leverage_factor: float) -> Dict[str, float]:
        """Calculate comprehensive trading risk metrics."""
        try:
            risk_metrics = {
                'leverage_factor': leverage_factor,
                'max_drawdown': dataset_meta_features.get('max_drawdown', 0),
                'sharpe_ratio': dataset_meta_features.get('sharpe_ratio', 0),
                'var_95': dataset_meta_features.get('var_95', 0),
                'leverage_risk': dataset_meta_features.get('leverage_risk', 0),
                'market_correlation': abs(dataset_meta_features.get('market_correlation', 0)),
                'calmar_ratio': dataset_meta_features.get('calmar_ratio', 0),
                'optimization_score': optimization_results.get('best_score', 0)
            }
            
            return risk_metrics
            
        except Exception as e:
            self.logger.warning(f"Risk metrics calculation failed: {e}")
            return {}
    
    def _calculate_meta_learning_confidence(self, 
                                          dataset_meta_features: Dict[str, float],
                                          model_type: str,
                                          market_regime: Optional[str]) -> float:
        """Calculate confidence in meta-learning predictions."""
        try:
            # Find similar datasets
            similar_datasets = self.history_db.find_similar_trading_datasets(
                dataset_meta_features, model_type, market_regime
            )
            
            if not similar_datasets:
                return 0.0
            
            # Confidence based on number and similarity of similar datasets
            avg_similarity = np.mean([d['similarity'] for d in similar_datasets])
            n_similar = len(similar_datasets)
            
            # Combine similarity and quantity
            confidence = avg_similarity * min(1.0, n_similar / 10.0)
            
            return float(confidence)
            
        except Exception as e:
            self.logger.warning(f"Meta-learning confidence calculation failed: {e}")
            return 0.0
    
    # Additional helper methods for search space adaptation
    def _adapt_search_space_from_history(self, 
                                       base_search_space: Dict[str, Any],
                                       similar_optimizations: List[Dict],
                                       dataset_meta_features: Dict[str, float]) -> Dict[str, Any]:
        """Adapt search space based on similar optimization history."""
        # Implementation would adapt parameter ranges based on similar datasets
        # This is a simplified version
        return base_search_space
    
    def _apply_regime_specific_adaptations(self, 
                                        search_space: Dict[str, Any],
                                        regime_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply regime-specific parameter adaptations."""
        # Implementation would adjust parameters based on regime-specific patterns
        return search_space
    
    def _apply_leverage_constraints(self, 
                                  search_space: Dict[str, Any],
                                  leverage_factor: float,
                                  dataset_meta_features: Dict[str, float]) -> Dict[str, Any]:
        """Apply leverage-specific constraints to search space."""
        # Implementation would adjust parameters based on leverage level
        return search_space
    
    def _apply_risk_aware_adaptations(self, 
                                    search_space: Dict[str, Any],
                                    dataset_meta_features: Dict[str, float]) -> Dict[str, Any]:
        """Apply risk-aware adaptations to search space."""
        # Implementation would adjust parameters based on risk characteristics
        return search_space
    
    def _optimize_search_space_for_budget(self, 
                                        search_space: Dict[str, Any],
                                        optimization_budget: int) -> Dict[str, Any]:
        """Optimize search space for given optimization budget."""
        # Implementation would adjust parameter ranges based on available trials
        return search_space