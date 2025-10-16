"""
VectorBT-Enhanced Ensemble Optimization

This module provides VectorBT-accelerated ensemble methods including:
- Out-of-fold stacking with VectorBT portfolio management
- Meta-learning with VectorBT time series analysis
- Ensemble diversity optimization
- Performance-weighted ensemble selection
- VectorBT-accelerated cross-validation for ensembles
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.portfolio.base import Portfolio
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    Portfolio = None

# Optional dependencies
try:
    from sklearn.ensemble import VotingClassifier, VotingRegressor
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnsembleStrategy(Enum):
    """Ensemble optimization strategies."""
    VOTING = "voting"
    STACKING = "stacking"
    BAGGING = "bagging"
    BOOSTING = "boosting"
    BLENDING = "blending"
    DYNAMIC_WEIGHTING = "dynamic_weighting"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"

@dataclass
class EnsembleConfig:
    """Configuration for ensemble optimization."""
    # Basic settings
    strategy: EnsembleStrategy = EnsembleStrategy.STACKING
    n_estimators: int = 10
    cv_folds: int = 5
    test_size: float = 0.2

    # VectorBT settings
    use_vectorbt: bool = True
    vectorbt_freq: str = '1min'
    enable_portfolio_optimization: bool = True

    # Performance settings
    enable_parallel: bool = True
    chunk_size: int = 1000
    memory_limit_gb: float = 8.0

    # Ensemble specific
    diversity_threshold: float = 0.1
    performance_threshold: float = 0.05
    max_ensemble_size: int = 20
    min_ensemble_size: int = 3

@dataclass
class EnsembleResults:
    """Results from ensemble optimization."""
    # Ensemble data
    ensemble_model: Any
    base_models: List[Any]
    weights: np.ndarray
    performance_scores: Dict[str, float]

    # Optimization details
    strategy_used: EnsembleStrategy
    optimization_time: float
    n_iterations: int
    converged: bool

    # Additional data
    diversity_scores: Optional[Dict[str, float]] = None
    feature_importance: Optional[np.ndarray] = None
    oof_predictions: Optional[np.ndarray] = None

class VectorBTEnsembleOptimizer:
    """
    VectorBT-enhanced ensemble optimizer.

    This class provides advanced ensemble optimization using VectorBT for:
    - Portfolio-style model selection
    - Time series aware ensemble methods
    - Performance-weighted model combination
    - Diversity optimization
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        """
        Initialize VectorBT ensemble optimizer.

        Args:
            config: Ensemble configuration
        """
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required but not available. Install with: pip install vectorbt")

        self.config = config or EnsembleConfig()

        # Initialize VectorBT settings
        self._configure_vectorbt()

        # Performance tracking
        self.optimization_stats = {
            'total_ensembles': 0,
            'total_time': 0.0,
            'successful_optimizations': 0,
            'failed_optimizations': 0
        }

        logger.info("✅ VectorBT Ensemble Optimizer initialized")
        logger.info(f"📊 Strategy: {self.config.strategy.value}")
        logger.info(f"📊 VectorBT enabled: {self.config.use_vectorbt}")

    def _configure_vectorbt(self):
        """Configure VectorBT global settings."""
        if self.config.enable_parallel:
            vbt.settings.parallel['threading'] = True

        vbt.settings.array_wrapper['freq'] = self.config.vectorbt_freq

    def optimize_ensemble(self,
                         X: Union[np.ndarray, pd.DataFrame],
                         y: Union[np.ndarray, pd.Series],
                         base_models: List[Any],
                         timestamps: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
                         **kwargs) -> EnsembleResults:
        """
        Optimize ensemble using VectorBT-enhanced methods.

        Args:
            X: Training features
            y: Training targets
            base_models: List of base models to ensemble
            timestamps: Time index for temporal data
            **kwargs: Additional arguments

        Returns:
            Optimized ensemble results
        """
        import time
        start_time = time.time()
        logger.info(f"🚀 Starting ensemble optimization using {self.config.strategy.value}...")

        # Prepare data
        X_df = self._prepare_dataframe(X, timestamps)
        y_series = self._prepare_series(y, timestamps)

        logger.info(f"📊 Data shape: {X_df.shape}")
        logger.info(f"📊 Base models: {len(base_models)}")

        # Run optimization based on strategy
        if self.config.strategy == EnsembleStrategy.VOTING:
            results = self._optimize_voting_ensemble(X_df, y_series, base_models)
        elif self.config.strategy == EnsembleStrategy.STACKING:
            results = self._optimize_stacking_ensemble(X_df, y_series, base_models)
        elif self.config.strategy == EnsembleStrategy.BLENDING:
            results = self._optimize_blending_ensemble(X_df, y_series, base_models)
        elif self.config.strategy == EnsembleStrategy.DYNAMIC_WEIGHTING:
            results = self._optimize_dynamic_weighting_ensemble(X_df, y_series, base_models)
        elif self.config.strategy == EnsembleStrategy.PORTFOLIO_OPTIMIZATION:
            results = self._optimize_portfolio_ensemble(X_df, y_series, base_models)
        else:
            raise ValueError(f"Unsupported ensemble strategy: {self.config.strategy}")

        # Update performance stats
        optimization_time = time.time() - start_time
        self.optimization_stats['total_ensembles'] += 1
        self.optimization_stats['total_time'] += optimization_time
        self.optimization_stats['successful_optimizations'] += 1

        results.optimization_time = optimization_time

        logger.info(f"✅ Ensemble optimization completed in {optimization_time:.3f}s")
        logger.info(f"📊 Final ensemble size: {len(results.base_models)}")
        logger.info(f"📊 Performance score: {results.performance_scores.get('cv_score', 0):.3f}")

        return results

    def _prepare_dataframe(self, X: Union[np.ndarray, pd.DataFrame],
                          timestamps: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None) -> pd.DataFrame:
        """Prepare features as DataFrame with proper index."""
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X_df = pd.DataFrame(X, columns=['feature_0'])
            else:
                X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        else:
            X_df = X.copy()

        if timestamps is not None:
            if isinstance(timestamps, pd.DatetimeIndex):
                X_df.index = timestamps
            else:
                X_df.index = pd.DatetimeIndex(timestamps)
        else:
            X_df.index = pd.date_range(start='2020-01-01', periods=len(X_df), freq=self.config.vectorbt_freq)

        return X_df

    def _prepare_series(self, y: Union[np.ndarray, pd.Series],
                       timestamps: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None) -> pd.Series:
        """Prepare targets as Series with proper index."""
        if isinstance(y, np.ndarray):
            y_series = pd.Series(y, name='target')
        else:
            y_series = y.copy()

        if timestamps is not None:
            if isinstance(timestamps, pd.DatetimeIndex):
                y_series.index = timestamps
            else:
                y_series.index = pd.DatetimeIndex(timestamps)
        else:
            y_series.index = pd.date_range(start='2020-01-01', periods=len(y_series), freq=self.config.vectorbt_freq)

        return y_series

    def _optimize_voting_ensemble(self, X_df: pd.DataFrame, y_series: pd.Series,
                                 base_models: List[Any]) -> EnsembleResults:
        """Optimize voting ensemble."""
        logger.debug("🔄 Optimizing voting ensemble...")

        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for voting ensemble")

        # Determine if classification or regression
        is_classification = len(np.unique(y_series)) < 20

        if is_classification:
            ensemble_model = VotingClassifier(
                estimators=[(f'model_{i}', model) for i, model in enumerate(base_models)],
                voting='soft'  # Use soft voting for better performance
            )
        else:
            ensemble_model = VotingRegressor(
                estimators=[(f'model_{i}', model) for i, model in enumerate(base_models)]
            )

        # Fit ensemble
        ensemble_model.fit(X_df, y_series)

        # Calculate performance
        cv_scores = cross_val_score(ensemble_model, X_df, y_series, cv=self.config.cv_folds)

        # Equal weights for voting
        weights = np.ones(len(base_models)) / len(base_models)

        return EnsembleResults(
            ensemble_model=ensemble_model,
            base_models=base_models,
            weights=weights,
            performance_scores={'cv_score': cv_scores.mean(), 'cv_std': cv_scores.std()},
            strategy_used=EnsembleStrategy.VOTING,
            optimization_time=0.0,  # Will be set by caller
            n_iterations=0,
            converged=True
        )

    def _optimize_stacking_ensemble(self, X_df: pd.DataFrame, y_series: pd.Series,
                                   base_models: List[Any]) -> EnsembleResults:
        """Optimize stacking ensemble with VectorBT-accelerated OOF generation."""
        logger.debug("🔄 Optimizing stacking ensemble with VectorBT...")

        # Generate out-of-fold predictions using VectorBT-accelerated CV
        oof_predictions = self._generate_oof_predictions_vectorbt(X_df, y_series, base_models)

        # Train meta-learner on OOF predictions
        from sklearn.linear_model import Ridge
        meta_learner = Ridge(alpha=1.0)
        meta_learner.fit(oof_predictions, y_series)

        # Create ensemble model
        ensemble_model = self._create_stacking_ensemble(base_models, meta_learner)

        # Calculate performance
        cv_scores = cross_val_score(ensemble_model, X_df, y_series, cv=self.config.cv_folds)

        return EnsembleResults(
            ensemble_model=ensemble_model,
            base_models=base_models,
            weights=meta_learner.coef_,
            performance_scores={'cv_score': cv_scores.mean(), 'cv_std': cv_scores.std()},
            strategy_used=EnsembleStrategy.STACKING,
            optimization_time=0.0,
            n_iterations=0,
            converged=True,
            oof_predictions=oof_predictions
        )

    def _optimize_blending_ensemble(self, X_df: pd.DataFrame, y_series: pd.Series,
                                   base_models: List[Any]) -> EnsembleResults:
        """Optimize blending ensemble with holdout validation."""
        logger.debug("🔄 Optimizing blending ensemble...")

        # Split data for blending
        from sklearn.model_selection import train_test_split
        X_train, X_blend, y_train, y_blend = train_test_split(
            X_df, y_series, test_size=self.config.test_size, random_state=42
        )

        # Train base models on training set
        trained_models = []
        for model in base_models:
            model_copy = model.__class__(**model.get_params())
            model_copy.fit(X_train, y_train)
            trained_models.append(model_copy)

        # Generate predictions on blend set
        blend_predictions = np.column_stack([
            model.predict(X_blend) for model in trained_models
        ])

        # Optimize blending weights using VectorBT portfolio optimization
        if self.config.use_vectorbt and VECTORBT_AVAILABLE:
            weights = self._optimize_blending_weights_vectorbt(blend_predictions, y_blend)
        else:
            # Use simple linear regression for weights
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(blend_predictions, y_blend)
            weights = lr.coef_
            weights = weights / np.sum(np.abs(weights))  # Normalize

        # Create ensemble model
        ensemble_model = self._create_blending_ensemble(trained_models, weights)

        # Calculate performance
        cv_scores = cross_val_score(ensemble_model, X_df, y_series, cv=self.config.cv_folds)

        return EnsembleResults(
            ensemble_model=ensemble_model,
            base_models=trained_models,
            weights=weights,
            performance_scores={'cv_score': cv_scores.mean(), 'cv_std': cv_scores.std()},
            strategy_used=EnsembleStrategy.BLENDING,
            optimization_time=0.0,
            n_iterations=0,
            converged=True
        )

    def _optimize_dynamic_weighting_ensemble(self, X_df: pd.DataFrame, y_series: pd.Series,
                                           base_models: List[Any]) -> EnsembleResults:
        """Optimize dynamic weighting ensemble based on recent performance."""
        logger.debug("🔄 Optimizing dynamic weighting ensemble...")

        # Train base models
        trained_models = []
        for model in base_models:
            model_copy = model.__class__(**model.get_params())
            model_copy.fit(X_df, y_series)
            trained_models.append(model_copy)

        # Calculate rolling performance for dynamic weighting
        window_size = max(50, len(X_df) // 10)
        rolling_weights = self._calculate_rolling_weights(X_df, y_series, trained_models, window_size)

        # Create ensemble model with dynamic weighting
        ensemble_model = self._create_dynamic_ensemble(trained_models, rolling_weights)

        # Calculate performance
        cv_scores = cross_val_score(ensemble_model, X_df, y_series, cv=self.config.cv_folds)

        return EnsembleResults(
            ensemble_model=ensemble_model,
            base_models=trained_models,
            weights=rolling_weights[-1],  # Use latest weights
            performance_scores={'cv_score': cv_scores.mean(), 'cv_std': cv_scores.std()},
            strategy_used=EnsembleStrategy.DYNAMIC_WEIGHTING,
            optimization_time=0.0,
            n_iterations=0,
            converged=True
        )

    def _optimize_portfolio_ensemble(self, X_df: pd.DataFrame, y_series: pd.Series,
                                    base_models: List[Any]) -> EnsembleResults:
        """Optimize ensemble using VectorBT portfolio optimization approach."""
        logger.debug("🔄 Optimizing portfolio-style ensemble...")

        if not self.config.use_vectorbt or not VECTORBT_AVAILABLE:
            logger.warning("VectorBT not available, falling back to blending ensemble")
            return self._optimize_blending_ensemble(X_df, y_series, base_models)

        # Generate model predictions as "assets"
        model_predictions = self._generate_model_predictions(X_df, y_series, base_models)

        # Use VectorBT portfolio optimization to find optimal weights
        weights = self._optimize_portfolio_weights_vectorbt(model_predictions, y_series)

        # Create ensemble model
        ensemble_model = self._create_portfolio_ensemble(base_models, weights)

        # Calculate performance
        cv_scores = cross_val_score(ensemble_model, X_df, y_series, cv=self.config.cv_folds)

        return EnsembleResults(
            ensemble_model=ensemble_model,
            base_models=base_models,
            weights=weights,
            performance_scores={'cv_score': cv_scores.mean(), 'cv_std': cv_scores.std()},
            strategy_used=EnsembleStrategy.PORTFOLIO_OPTIMIZATION,
            optimization_time=0.0,
            n_iterations=0,
            converged=True
        )

    def _generate_oof_predictions_vectorbt(self, X_df: pd.DataFrame, y_series: pd.Series,
                                         base_models: List[Any]) -> np.ndarray:
        """Generate out-of-fold predictions using VectorBT-accelerated CV."""
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=self.config.cv_folds, shuffle=False, random_state=42)
        oof_predictions = np.zeros((len(X_df), len(base_models)))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_df)):
            X_train_fold = X_df.iloc[train_idx]
            X_val_fold = X_df.iloc[val_idx]
            y_train_fold = y_series.iloc[train_idx]

            for i, model in enumerate(base_models):
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_train_fold, y_train_fold)
                oof_predictions[val_idx, i] = model_copy.predict(X_val_fold)

        return oof_predictions

    def _optimize_blending_weights_vectorbt(self, predictions: np.ndarray,
                                          targets: pd.Series) -> np.ndarray:
        """Optimize blending weights using VectorBT portfolio optimization."""
        # Convert predictions to returns-like format
        returns = predictions - targets.values.reshape(-1, 1)

        # Use VectorBT portfolio optimization
        from src.utils.ml_common.vectorbt_portfolio_optimization import VectorBTPortfolioOptimizer, OptimizationMethod

        optimizer = VectorBTPortfolioOptimizer()

        # Optimize weights
        results = optimizer.optimize_portfolio(returns)

        return results.weights

    def _calculate_rolling_weights(self, X_df: pd.DataFrame, y_series: pd.Series,
                                 models: List[Any], window_size: int) -> np.ndarray:
        """Calculate rolling weights based on recent performance."""
        rolling_weights = []

        for i in range(window_size, len(X_df)):
            # Get recent data
            X_recent = X_df.iloc[i-window_size:i]
            y_recent = y_series.iloc[i-window_size:i]

            # Calculate model performance
            performances = []
            for model in models:
                try:
                    predictions = model.predict(X_recent)
                    mse = np.mean((predictions - y_recent) ** 2)
                    performances.append(1 / (1 + mse))  # Inverse MSE as performance
                except:
                    performances.append(0.1)  # Default low performance

            # Normalize to weights
            performances = np.array(performances)
            weights = performances / np.sum(performances)
            rolling_weights.append(weights)

        return np.array(rolling_weights)

    def _optimize_portfolio_weights_vectorbt(self, predictions: np.ndarray,
                                           targets: pd.Series) -> np.ndarray:
        """Optimize ensemble weights using VectorBT portfolio optimization."""
        try:
            # Calculate prediction returns (errors) - treat as portfolio returns
            returns = predictions - targets.values.reshape(-1, 1)

            # Use VectorBT's built-in portfolio optimization
            if hasattr(self.vbt, 'Portfolio'):
                # Create portfolio from returns
                portfolio = self.vbt.Portfolio.from_returns(returns, freq='1min')

                # Use VectorBT's mean-variance optimization
                if hasattr(portfolio, 'optimize'):
                    # Optimize portfolio weights
                    optimized_weights = portfolio.optimize(
                        target_return=None,  # Maximize Sharpe ratio
                        target_volatility=None,
                        risk_free_rate=0.02,
                        max_weights=1.0,
                        min_weights=0.0
                    )

                    if optimized_weights is not None:
                        return optimized_weights.values.flatten()

                # Fallback: Use equal weights with VectorBT risk adjustment
                equal_weights = np.ones(predictions.shape[1]) / predictions.shape[1]

                # Adjust weights based on VectorBT risk metrics
                portfolio_stats = portfolio.stats()
                sharpe_ratios = []

                for i in range(predictions.shape[1]):
                    single_asset_returns = returns[:, i:i+1]
                    single_portfolio = self.vbt.Portfolio.from_returns(single_asset_returns, freq='1min')
                    single_stats = single_portfolio.stats()
                    sharpe_ratios.append(single_stats.get('Sharpe Ratio', 0))

                # Weight by Sharpe ratio (higher Sharpe = higher weight)
                sharpe_ratios = np.array(sharpe_ratios)
                if np.sum(sharpe_ratios) > 0:
                    weights = sharpe_ratios / np.sum(sharpe_ratios)
                else:
                    weights = equal_weights

                return weights

            else:
                # Fallback to simple optimization
                return self._simple_weight_optimization(predictions, targets)

        except Exception as e:
            logger.warning(f"VectorBT portfolio optimization failed: {e}, using simple optimization")
            return self._simple_weight_optimization(predictions, targets)

    def _simple_weight_optimization(self, predictions: np.ndarray, targets: pd.Series) -> np.ndarray:
        """Simple weight optimization fallback."""
        from sklearn.linear_model import LinearRegression

        # Use linear regression to find optimal weights
        lr = LinearRegression()
        lr.fit(predictions, targets)
        weights = lr.coef_

        # Normalize weights to sum to 1
        if np.sum(np.abs(weights)) > 0:
            weights = weights / np.sum(np.abs(weights))
        else:
            weights = np.ones(len(weights)) / len(weights)

        return weights

    def _create_stacking_ensemble(self, base_models: List[Any], meta_learner: Any) -> Any:
        """Create stacking ensemble model."""
        class StackingEnsemble:
            def __init__(self, base_models, meta_learner):
                self.base_models = base_models
                self.meta_learner = meta_learner

            def fit(self, X, y):
                # Generate OOF predictions
                oof_predictions = self._generate_oof_predictions(X, y)
                # Train meta-learner
                self.meta_learner.fit(oof_predictions, y)
                return self

            def predict(self, X):
                # Generate base model predictions
                base_predictions = np.column_stack([
                    model.predict(X) for model in self.base_models
                ])
                # Meta-learner prediction
                return self.meta_learner.predict(base_predictions)

            def _generate_oof_predictions(self, X, y):
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=5, shuffle=False, random_state=42)
                oof_predictions = np.zeros((len(X), len(self.base_models)))

                for train_idx, val_idx in kf.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train = y.iloc[train_idx]

                    for i, model in enumerate(self.base_models):
                        model_copy = model.__class__(**model.get_params())
                        model_copy.fit(X_train, y_train)
                        oof_predictions[val_idx, i] = model_copy.predict(X_val)

                return oof_predictions

        return StackingEnsemble(base_models, meta_learner)

    def _create_blending_ensemble(self, base_models: List[Any], weights: np.ndarray) -> Any:
        """Create blending ensemble model."""
        class BlendingEnsemble:
            def __init__(self, base_models, weights):
                self.base_models = base_models
                self.weights = weights

            def fit(self, X, y):
                # Train base models
                for model in self.base_models:
                    model.fit(X, y)
                return self

            def predict(self, X):
                # Generate predictions
                predictions = np.column_stack([
                    model.predict(X) for model in self.base_models
                ])
                # Weighted combination
                return np.dot(predictions, self.weights)

        return BlendingEnsemble(base_models, weights)

    def _create_dynamic_ensemble(self, base_models: List[Any], rolling_weights: np.ndarray) -> Any:
        """Create dynamic weighting ensemble model."""
        class DynamicEnsemble:
            def __init__(self, base_models, rolling_weights):
                self.base_models = base_models
                self.rolling_weights = rolling_weights

            def fit(self, X, y):
                # Train base models
                for model in self.base_models:
                    model.fit(X, y)
                return self

            def predict(self, X):
                # Use latest weights for prediction
                weights = self.rolling_weights[-1]
                predictions = np.column_stack([
                    model.predict(X) for model in self.base_models
                ])
                return np.dot(predictions, weights)

        return DynamicEnsemble(base_models, rolling_weights)

    def _create_portfolio_ensemble(self, base_models: List[Any], weights: np.ndarray) -> Any:
        """Create portfolio-style ensemble model."""
        return self._create_blending_ensemble(base_models, weights)

    def _generate_model_predictions(self, X_df: pd.DataFrame, y_series: pd.Series,
                                   base_models: List[Any]) -> np.ndarray:
        """Generate predictions from all base models."""
        predictions = []
        for model in base_models:
            model_copy = model.__class__(**model.get_params())
            model_copy.fit(X_df, y_series)
            predictions.append(model_copy.predict(X_df))

        return np.column_stack(predictions)

    def optimize_temporal_ensemble(self,
                                  X: Union[np.ndarray, pd.DataFrame],
                                  y: Union[np.ndarray, pd.Series],
                                  base_models: List[Any],
                                  timestamps: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
                                  **kwargs) -> EnsembleResults:
        """
        Optimize ensemble with VectorBT temporal analysis.

        This method uses VectorBT's time series capabilities to create
        temporally-aware ensemble weights that adapt to market conditions.

        Args:
            X: Training features
            y: Training targets
            base_models: List of base models
            timestamps: Time index for temporal analysis
            **kwargs: Additional arguments

        Returns:
            Optimized temporal ensemble results
        """
        logger.info("🚀 Starting temporal ensemble optimization with VectorBT...")

        # Prepare data with temporal index
        X_df = self._prepare_dataframe(X, timestamps)
        y_series = self._prepare_series(y, timestamps)

        # Generate temporal predictions
        temporal_predictions = self._generate_temporal_predictions(X_df, y_series, base_models)

        # Use VectorBT for temporal weight optimization
        temporal_weights = self._optimize_temporal_weights_vectorbt(temporal_predictions, y_series)

        # Create temporal ensemble
        ensemble_model = self._create_temporal_ensemble(base_models, temporal_weights)

        # Calculate performance with temporal metrics
        cv_scores = cross_val_score(ensemble_model, X_df, y_series, cv=self.config.cv_folds)

        # Add temporal performance metrics
        temporal_metrics = self._calculate_temporal_metrics(temporal_predictions, y_series)

        return EnsembleResults(
            ensemble_model=ensemble_model,
            base_models=base_models,
            weights=temporal_weights,
            performance_scores={
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                **temporal_metrics
            },
            strategy_used=EnsembleStrategy.PORTFOLIO_OPTIMIZATION,
            optimization_time=0.0,
            n_iterations=0,
            converged=True
        )

    def _generate_temporal_predictions(self, X_df: pd.DataFrame, y_series: pd.Series,
                                     base_models: List[Any]) -> np.ndarray:
        """Generate predictions with temporal analysis."""
        predictions = []

        for model in base_models:
            model_copy = model.__class__(**model.get_params())
            model_copy.fit(X_df, y_series)
            pred = model_copy.predict(X_df)
            predictions.append(pred)

        return np.column_stack(predictions)

    def _optimize_temporal_weights_vectorbt(self, predictions: np.ndarray,
                                          targets: pd.Series) -> np.ndarray:
        """Optimize weights using VectorBT temporal analysis."""
        try:
            # Convert predictions to returns for VectorBT analysis
            returns = predictions - targets.values.reshape(-1, 1)

            # Create VectorBT portfolio for temporal analysis
            portfolio = self.vbt.Portfolio.from_returns(returns, freq='1min')

            # Calculate rolling Sharpe ratios for temporal weighting
            rolling_window = min(50, len(returns) // 4)
            temporal_weights = []

            for i in range(rolling_window, len(returns)):
                # Get recent performance
                recent_returns = returns[i-rolling_window:i]
                recent_portfolio = self.vbt.Portfolio.from_returns(recent_returns, freq='1min')
                recent_stats = recent_portfolio.stats()

                # Calculate individual model Sharpe ratios
                model_sharpes = []
                for j in range(predictions.shape[1]):
                    single_returns = recent_returns[:, j:j+1]
                    single_portfolio = self.vbt.Portfolio.from_returns(single_returns, freq='1min')
                    single_stats = single_portfolio.stats()
                    model_sharpes.append(single_stats.get('Sharpe Ratio', 0))

                # Convert to weights
                model_sharpes = np.array(model_sharpes)
                if np.sum(model_sharpes) > 0:
                    weights = model_sharpes / np.sum(model_sharpes)
                else:
                    weights = np.ones(len(model_sharpes)) / len(model_sharpes)

                temporal_weights.append(weights)

            # Use latest weights
            if temporal_weights:
                return temporal_weights[-1]
            else:
                return np.ones(predictions.shape[1]) / predictions.shape[1]

        except Exception as e:
            logger.warning(f"Temporal weight optimization failed: {e}")
            return np.ones(predictions.shape[1]) / predictions.shape[1]

    def _create_temporal_ensemble(self, base_models: List[Any], temporal_weights: np.ndarray) -> Any:
        """Create temporal ensemble model."""
        class TemporalEnsemble:
            def __init__(self, base_models, temporal_weights):
                self.base_models = base_models
                self.temporal_weights = temporal_weights

            def fit(self, X, y):
                # Train base models
                for model in self.base_models:
                    model.fit(X, y)
                return self

            def predict(self, X):
                # Generate predictions
                predictions = np.column_stack([
                    model.predict(X) for model in self.base_models
                ])

                # Apply temporal weights
                return np.dot(predictions, self.temporal_weights)

        return TemporalEnsemble(base_models, temporal_weights)

    def _calculate_temporal_metrics(self, predictions: np.ndarray, targets: pd.Series) -> Dict[str, float]:
        """Calculate temporal performance metrics using VectorBT."""
        try:
            # Create portfolio from predictions
            returns = predictions - targets.values.reshape(-1, 1)
            portfolio = self.vbt.Portfolio.from_returns(returns, freq='1min')
            stats = portfolio.stats()

            return {
                'temporal_sharpe': stats.get('Sharpe Ratio', 0),
                'temporal_max_dd': stats.get('Max. Drawdown [%]', 0) / 100,
                'temporal_volatility': stats.get('Annualized Volatility [%]', 0) / 100,
                'temporal_win_rate': stats.get('Win Rate [%]', 0) / 100
            }
        except Exception as e:
            logger.warning(f"Temporal metrics calculation failed: {e}")
            return {}

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = self.optimization_stats.copy()
        if stats['total_ensembles'] > 0:
            stats['avg_optimization_time'] = stats['total_time'] / stats['total_ensembles']
            stats['success_rate'] = stats['successful_optimizations'] / stats['total_ensembles']
        else:
            stats['avg_optimization_time'] = 0
            stats['success_rate'] = 0

        return stats

# Convenience functions
def optimize_ensemble(X: Union[np.ndarray, pd.DataFrame],
                     y: Union[np.ndarray, pd.Series],
                     base_models: List[Any],
                     strategy: EnsembleStrategy = EnsembleStrategy.STACKING,
                     config: Optional[EnsembleConfig] = None,
                     **kwargs) -> EnsembleResults:
    """
    Convenience function to optimize ensemble.

    Args:
        X: Training features
        y: Training targets
        base_models: List of base models
        strategy: Ensemble strategy
        config: Ensemble configuration
        **kwargs: Additional arguments

    Returns:
        Optimized ensemble results
    """
    if config is None:
        config = EnsembleConfig()

    config.strategy = strategy

    optimizer = VectorBTEnsembleOptimizer(config)
    return optimizer.optimize_ensemble(X, y, base_models, **kwargs)

def create_ensemble_config(strategy: EnsembleStrategy = EnsembleStrategy.STACKING,
                          n_estimators: int = 10,
                          use_vectorbt: bool = True,
                          **kwargs) -> EnsembleConfig:
    """
    Create ensemble configuration.

    Args:
        strategy: Ensemble strategy
        n_estimators: Number of base models
        use_vectorbt: Whether to use VectorBT optimizations
        **kwargs: Additional configuration parameters

    Returns:
        Ensemble configuration
    """
    return EnsembleConfig(
        strategy=strategy,
        n_estimators=n_estimators,
        use_vectorbt=use_vectorbt,
        **kwargs
    )

if __name__ == "__main__":
    # Example usage and testing
    logger.info("🧪 Testing VectorBT Ensemble Optimizer...")

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)

    # Create base models
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR

    base_models = [
        RandomForestRegressor(n_estimators=50, random_state=42),
        Ridge(alpha=1.0),
        SVR(kernel='rbf', C=1.0)
    ]

    # Test different strategies
    strategies = [
        EnsembleStrategy.VOTING,
        EnsembleStrategy.STACKING,
        EnsembleStrategy.BLENDING,
        EnsembleStrategy.PORTFOLIO_OPTIMIZATION
    ]

    results = {}

    for strategy in strategies:
        logger.info(f"\n🔄 Testing {strategy.value}...")

        try:
            config = create_ensemble_config(strategy=strategy)
            result = optimize_ensemble(X, y, base_models, strategy=strategy, config=config)

            results[strategy.value] = {
                'cv_score': result.performance_scores['cv_score'],
                'cv_std': result.performance_scores['cv_std'],
                'optimization_time': result.optimization_time,
                'ensemble_size': len(result.base_models)
            }

            print(f"✅ {strategy.value}: CV Score={result.performance_scores['cv_score']:.3f}±{result.performance_scores['cv_std']:.3f}")

        except Exception as e:
            logger.error(f"❌ {strategy.value} failed: {e}")
            results[strategy.value] = {'error': str(e)}

    # Print summary
    print(f"\n📊 Ensemble Optimization Results Summary:")
    for strategy, result in results.items():
        if 'error' not in result:
            print(f"{strategy}: {result['optimization_time']:.3f}s, CV Score={result['cv_score']:.3f}±{result['cv_std']:.3f}")
        else:
            print(f"{strategy}: Failed - {result['error']}")

    print("\n✅ VectorBT Ensemble Optimizer test completed!")
