"""
Automated Feature Engineering for Market Regime Discovery

This module implements advanced ML techniques for automated feature engineering:
1. Automated feature synthesis using genetic programming
2. Deep feature interactions discovery
3. Polynomial and interaction feature generation
4. Time-series specific feature engineering
5. Domain-aware feature construction for financial markets
6. Feature selection optimization using ML

Key Capabilities:
- Genetic Programming for feature evolution
- Neural network-based feature synthesis
- Automated interaction detection
- Time-aware feature engineering
- Financial domain-specific transformations
- Multi-objective feature optimization (performance vs complexity)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json
import warnings
from abc import ABC, abstractmethod
import itertools
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline

try:
    import gplearn
    from gplearn.genetic import SymbolicRegressor, SymbolicClassifier
    from gplearn.functions import make_function
    GPLEARN_AVAILABLE = True
except ImportError:
    GPLEARN_AVAILABLE = False

try:
    import featuretools as ft
    FEATURETOOLS_AVAILABLE = True
except ImportError:
    FEATURETOOLS_AVAILABLE = False

from src.utils.logger import system_logger


class FeatureEngineeringMethod(Enum):
    """Methods for automated feature engineering."""
    GENETIC_PROGRAMMING = "genetic_programming"
    POLYNOMIAL_FEATURES = "polynomial_features"
    INTERACTION_FEATURES = "interaction_features"
    TIME_SERIES_FEATURES = "time_series_features"
    NEURAL_FEATURE_SYNTHESIS = "neural_synthesis"
    DOMAIN_SPECIFIC_FEATURES = "domain_specific"
    RECURSIVE_FEATURE_ELIMINATION = "rfe"
    MUTUAL_INFORMATION_SELECTION = "mutual_info"
    ENSEMBLE_FEATURE_SELECTION = "ensemble_selection"


class FinancialTransform(Enum):
    """Financial domain-specific transformations."""
    LOG_RETURNS = "log_returns"
    ROLLING_STATISTICS = "rolling_stats"
    VOLATILITY_FEATURES = "volatility"
    MOMENTUM_FEATURES = "momentum"
    MEAN_REVERSION_FEATURES = "mean_reversion"
    REGIME_INDICATORS = "regime_indicators"
    CROSS_ASSET_RATIOS = "cross_asset_ratios"
    TECHNICAL_INDICATORS = "technical_indicators"


@dataclass
class AutoFeatureConfig:
    """Configuration for automated feature engineering."""
    # Genetic Programming parameters
    population_size: int = 1000
    generations: int = 20
    tournament_size: int = 20
    stopping_criteria: float = 0.01
    const_range: Tuple[float, float] = (-1.0, 1.0)
    init_depth: Tuple[int, int] = (2, 6)
    init_method: str = 'half and half'
    
    # Polynomial features
    poly_degree: int = 2
    poly_interaction_only: bool = True
    poly_include_bias: bool = False
    
    # Time series features
    rolling_windows: List[int] = None
    lag_features: List[int] = None
    seasonal_periods: List[int] = None
    
    # Neural synthesis
    synthesis_epochs: int = 100
    synthesis_lr: float = 0.001
    synthesis_hidden_dims: List[int] = None
    
    # Feature selection
    selection_k: int = 50
    selection_method: str = "mutual_info"
    cv_folds: int = 5
    
    # Financial parameters
    financial_transforms: List[str] = None
    volatility_windows: List[int] = None
    momentum_windows: List[int] = None
    
    # Optimization parameters
    max_features: int = 1000
    feature_importance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    
    # Performance parameters
    n_jobs: int = -1
    random_state: int = 42
    verbose: bool = True
    
    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [5, 10, 20, 50]
        if self.lag_features is None:
            self.lag_features = [1, 2, 3, 5, 10]
        if self.seasonal_periods is None:
            self.seasonal_periods = [5, 21, 63, 252]  # Daily, weekly, monthly, yearly
        if self.synthesis_hidden_dims is None:
            self.synthesis_hidden_dims = [128, 64, 32]
        if self.financial_transforms is None:
            self.financial_transforms = ["log_returns", "rolling_stats", "volatility", "momentum"]
        if self.volatility_windows is None:
            self.volatility_windows = [5, 10, 20, 50]
        if self.momentum_windows is None:
            self.momentum_windows = [5, 10, 20, 50]


class GeneticFeatureEngineer:
    """Genetic Programming for automated feature discovery."""
    
    def __init__(self, config: AutoFeatureConfig):
        self.config = config
        self.logger = system_logger.getChild('GeneticFeatureEngineer')
        self.evolved_features = []
        self.feature_scores = []
    
    def engineer_features(
        self, 
        X: pd.DataFrame, 
        y: np.ndarray,
        feature_names: List[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Engineer features using genetic programming."""
        
        if not GPLEARN_AVAILABLE:
            self.logger.warning("gplearn not available, skipping genetic programming")
            return X, {'error': 'gplearn not available'}
        
        self.logger.info("🧬 Starting genetic programming feature evolution")
        
        if feature_names is None:
            feature_names = list(X.columns)
        
        # Determine if classification or regression
        unique_targets = len(np.unique(y))
        is_classification = unique_targets < min(len(y) * 0.1, 50)
        
        # Define custom functions for financial markets
        financial_functions = self._create_financial_functions()
        
        if is_classification:
            gp_model = SymbolicClassifier(
                population_size=self.config.population_size,
                generations=self.config.generations,
                tournament_size=self.config.tournament_size,
                stopping_criteria=self.config.stopping_criteria,
                const_range=self.config.const_range,
                init_depth=self.config.init_depth,
                init_method=self.config.init_method,
                function_set=financial_functions,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose
            )
        else:
            gp_model = SymbolicRegressor(
                population_size=self.config.population_size,
                generations=self.config.generations,
                tournament_size=self.config.tournament_size,
                stopping_criteria=self.config.stopping_criteria,
                const_range=self.config.const_range,
                init_depth=self.config.init_depth,
                init_method=self.config.init_method,
                function_set=financial_functions,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose
            )
        
        # Fit genetic programming model
        try:
            gp_model.fit(X.values, y)
            
            # Extract evolved features from the population
            evolved_features_data = []
            feature_expressions = []
            
            # Get best programs from final population
            best_programs = sorted(gp_model._programs[-1], 
                                 key=lambda x: x.fitness_, reverse=True)
            
            for i, program in enumerate(best_programs[:self.config.selection_k]):
                if program.fitness_ > self.config.feature_importance_threshold:
                    # Transform data using this program
                    feature_values = program.execute(X.values)
                    
                    # Handle infinite or NaN values
                    if np.any(np.isfinite(feature_values)):
                        feature_values = np.nan_to_num(feature_values, 
                                                     nan=0.0, posinf=1e6, neginf=-1e6)
                        evolved_features_data.append(feature_values)
                        feature_expressions.append(str(program))
            
            # Create DataFrame with evolved features
            if evolved_features_data:
                evolved_df = pd.DataFrame(
                    np.column_stack(evolved_features_data),
                    columns=[f'gp_feature_{i}' for i in range(len(evolved_features_data))],
                    index=X.index
                )
                
                # Combine with original features
                enhanced_features = pd.concat([X, evolved_df], axis=1)
            else:
                enhanced_features = X.copy()
            
            metadata = {
                'method': 'genetic_programming',
                'n_evolved_features': len(evolved_features_data),
                'best_fitness': float(gp_model._program.fitness_) if hasattr(gp_model, '_program') else 0.0,
                'generations_completed': gp_model.generations,
                'feature_expressions': feature_expressions[:10]  # Top 10 expressions
            }
            
            self.logger.info(f"✅ Evolved {len(evolved_features_data)} new features")
            
        except Exception as e:
            self.logger.error(f"❌ Genetic programming failed: {e}")
            enhanced_features = X.copy()
            metadata = {'error': str(e)}
        
        return enhanced_features, metadata
    
    def _create_financial_functions(self):
        """Create custom functions for financial feature engineering."""
        
        def safe_divide(x1, x2):
            """Safe division avoiding division by zero."""
            return np.where(np.abs(x2) > 1e-6, np.divide(x1, x2), 0.0)
        
        def rolling_mean(x1, window=5):
            """Rolling mean approximation."""
            # Simple approximation for GP
            return np.convolve(x1, np.ones(window)/window, mode='same')
        
        def momentum(x1, window=5):
            """Price momentum."""
            if len(x1) < window:
                return np.zeros_like(x1)
            return x1 - np.roll(x1, window)
        
        def volatility(x1, window=5):
            """Rolling volatility approximation."""
            if len(x1) < window:
                return np.ones_like(x1)
            returns = np.diff(x1, prepend=x1[0])
            return np.convolve(returns**2, np.ones(window)/window, mode='same')
        
        # Register custom functions
        safe_div = make_function(function=safe_divide, name='safe_div', arity=2)
        momentum_func = make_function(function=momentum, name='momentum', arity=1)
        volatility_func = make_function(function=volatility, name='volatility', arity=1)
        
        return ['add', 'sub', 'mul', safe_div, 'sqrt', 'log', 'abs', 
                'neg', 'inv', momentum_func, volatility_func, 'sin', 'cos']


class PolynomialFeatureEngineer:
    """Polynomial and interaction feature engineering."""
    
    def __init__(self, config: AutoFeatureConfig):
        self.config = config
        self.logger = system_logger.getChild('PolynomialFeatureEngineer')
        self.poly_transformer = None
    
    def engineer_features(
        self, 
        X: pd.DataFrame, 
        y: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create polynomial and interaction features."""
        
        self.logger.info(f"🔢 Creating polynomial features (degree={self.config.poly_degree})")
        
        # Limit features to prevent explosion
        if X.shape[1] > 50:
            # Select top features first
            selector = SelectKBest(f_classif, k=50)
            X_selected = pd.DataFrame(
                selector.fit_transform(X, y) if y is not None else X.iloc[:, :50],
                columns=X.columns[:50] if y is None else X.columns[selector.get_support()],
                index=X.index
            )
        else:
            X_selected = X.copy()
        
        # Create polynomial features
        self.poly_transformer = PolynomialFeatures(
            degree=self.config.poly_degree,
            interaction_only=self.config.poly_interaction_only,
            include_bias=self.config.poly_include_bias
        )
        
        try:
            poly_features = self.poly_transformer.fit_transform(X_selected.fillna(0))
            
            # Create feature names
            feature_names = self.poly_transformer.get_feature_names_out(X_selected.columns)
            
            # Create DataFrame
            poly_df = pd.DataFrame(
                poly_features,
                columns=feature_names,
                index=X.index
            )
            
            # Remove original features to avoid duplication
            original_features = set(X_selected.columns)
            new_features = [col for col in poly_df.columns if col not in original_features]
            
            enhanced_features = pd.concat([
                X, 
                poly_df[new_features]
            ], axis=1)
            
            metadata = {
                'method': 'polynomial_features',
                'degree': self.config.poly_degree,
                'n_original_features': X_selected.shape[1],
                'n_polynomial_features': len(new_features),
                'interaction_only': self.config.poly_interaction_only
            }
            
            self.logger.info(f"✅ Created {len(new_features)} polynomial features")
            
        except Exception as e:
            self.logger.error(f"❌ Polynomial feature creation failed: {e}")
            enhanced_features = X.copy()
            metadata = {'error': str(e)}
        
        return enhanced_features, metadata


class TimeSeriesFeatureEngineer:
    """Time series specific feature engineering."""
    
    def __init__(self, config: AutoFeatureConfig):
        self.config = config
        self.logger = system_logger.getChild('TimeSeriesFeatureEngineer')
    
    def engineer_features(
        self, 
        X: pd.DataFrame, 
        y: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create time series specific features."""
        
        self.logger.info("📈 Creating time series features")
        
        enhanced_features = X.copy()
        feature_count = 0
        
        try:
            # Lag features
            for col in X.select_dtypes(include=[np.number]).columns:
                for lag in self.config.lag_features:
                    lag_feature = X[col].shift(lag)
                    enhanced_features[f'{col}_lag_{lag}'] = lag_feature
                    feature_count += 1
            
            # Rolling statistics
            for col in X.select_dtypes(include=[np.number]).columns:
                for window in self.config.rolling_windows:
                    # Rolling mean
                    enhanced_features[f'{col}_rolling_mean_{window}'] = X[col].rolling(window).mean()
                    
                    # Rolling std
                    enhanced_features[f'{col}_rolling_std_{window}'] = X[col].rolling(window).std()
                    
                    # Rolling min/max
                    enhanced_features[f'{col}_rolling_min_{window}'] = X[col].rolling(window).min()
                    enhanced_features[f'{col}_rolling_max_{window}'] = X[col].rolling(window).max()
                    
                    # Rolling skew/kurtosis
                    enhanced_features[f'{col}_rolling_skew_{window}'] = X[col].rolling(window).skew()
                    enhanced_features[f'{col}_rolling_kurtosis_{window}'] = X[col].rolling(window).kurtosis()
                    
                    feature_count += 6
            
            # Difference features
            for col in X.select_dtypes(include=[np.number]).columns:
                # First difference
                enhanced_features[f'{col}_diff_1'] = X[col].diff(1)
                
                # Second difference
                enhanced_features[f'{col}_diff_2'] = X[col].diff(2)
                
                # Percentage change
                enhanced_features[f'{col}_pct_change'] = X[col].pct_change()
                
                feature_count += 3
            
            # Seasonal features (if datetime index)
            if isinstance(X.index, pd.DatetimeIndex):
                enhanced_features['hour'] = X.index.hour
                enhanced_features['day_of_week'] = X.index.dayofweek
                enhanced_features['day_of_month'] = X.index.day
                enhanced_features['month'] = X.index.month
                enhanced_features['quarter'] = X.index.quarter
                
                # Cyclical encoding
                enhanced_features['hour_sin'] = np.sin(2 * np.pi * X.index.hour / 24)
                enhanced_features['hour_cos'] = np.cos(2 * np.pi * X.index.hour / 24)
                enhanced_features['day_sin'] = np.sin(2 * np.pi * X.index.dayofweek / 7)
                enhanced_features['day_cos'] = np.cos(2 * np.pi * X.index.dayofweek / 7)
                
                feature_count += 9
            
            # Autocorrelation features
            for col in X.select_dtypes(include=[np.number]).columns:
                for lag in [1, 5, 10]:
                    autocorr = X[col].rolling(50).apply(
                        lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
                    )
                    enhanced_features[f'{col}_autocorr_{lag}'] = autocorr
                    feature_count += 1
            
            metadata = {
                'method': 'time_series_features',
                'n_new_features': feature_count,
                'lag_features': self.config.lag_features,
                'rolling_windows': self.config.rolling_windows,
                'has_datetime_index': isinstance(X.index, pd.DatetimeIndex)
            }
            
            self.logger.info(f"✅ Created {feature_count} time series features")
            
        except Exception as e:
            self.logger.error(f"❌ Time series feature creation failed: {e}")
            enhanced_features = X.copy()
            metadata = {'error': str(e)}
        
        return enhanced_features, metadata


class FinancialFeatureEngineer:
    """Financial domain-specific feature engineering."""
    
    def __init__(self, config: AutoFeatureConfig):
        self.config = config
        self.logger = system_logger.getChild('FinancialFeatureEngineer')
    
    def engineer_features(
        self, 
        X: pd.DataFrame, 
        price_columns: Optional[List[str]] = None,
        volume_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Create financial domain-specific features."""
        
        self.logger.info("💰 Creating financial domain features")
        
        enhanced_features = X.copy()
        feature_count = 0
        
        # Auto-detect price and volume columns if not provided
        if price_columns is None:
            price_columns = [col for col in X.columns 
                           if any(keyword in col.lower() 
                                 for keyword in ['price', 'close', 'open', 'high', 'low'])]
        
        if volume_columns is None:
            volume_columns = [col for col in X.columns 
                            if any(keyword in col.lower() 
                                  for keyword in ['volume', 'vol', 'qty'])]
        
        try:
            # Price-based features
            for col in price_columns:
                if col in X.columns:
                    # Log returns
                    enhanced_features[f'{col}_log_return'] = np.log(X[col] / X[col].shift(1))
                    
                    # Volatility (rolling std of returns)
                    for window in self.config.volatility_windows:
                        returns = X[col].pct_change()
                        vol = returns.rolling(window).std()
                        enhanced_features[f'{col}_volatility_{window}'] = vol
                        feature_count += 1
                    
                    # Momentum indicators
                    for window in self.config.momentum_windows:
                        momentum = X[col] / X[col].shift(window) - 1
                        enhanced_features[f'{col}_momentum_{window}'] = momentum
                        feature_count += 1
                    
                    # Moving averages
                    for window in self.config.rolling_windows:
                        ma = X[col].rolling(window).mean()
                        enhanced_features[f'{col}_ma_{window}'] = ma
                        
                        # Price relative to MA
                        enhanced_features[f'{col}_price_to_ma_{window}'] = X[col] / ma - 1
                        feature_count += 2
                    
                    # Bollinger Bands
                    for window in [20, 50]:
                        ma = X[col].rolling(window).mean()
                        std = X[col].rolling(window).std()
                        
                        upper_band = ma + 2 * std
                        lower_band = ma - 2 * std
                        
                        enhanced_features[f'{col}_bb_upper_{window}'] = upper_band
                        enhanced_features[f'{col}_bb_lower_{window}'] = lower_band
                        enhanced_features[f'{col}_bb_position_{window}'] = (X[col] - lower_band) / (upper_band - lower_band)
                        feature_count += 3
                    
                    # RSI (Relative Strength Index)
                    delta = X[col].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    enhanced_features[f'{col}_rsi'] = rsi
                    feature_count += 1
            
            # Volume-based features
            for vol_col in volume_columns:
                if vol_col in X.columns:
                    # Volume moving averages
                    for window in self.config.rolling_windows:
                        vol_ma = X[vol_col].rolling(window).mean()
                        enhanced_features[f'{vol_col}_ma_{window}'] = vol_ma
                        
                        # Volume relative to average
                        enhanced_features[f'{vol_col}_relative_{window}'] = X[vol_col] / vol_ma
                        feature_count += 2
                    
                    # Volume volatility
                    vol_volatility = X[vol_col].rolling(20).std()
                    enhanced_features[f'{vol_col}_volatility'] = vol_volatility
                    feature_count += 1
            
            # Price-Volume interactions
            if price_columns and volume_columns:
                for price_col in price_columns[:2]:  # Limit to avoid explosion
                    for vol_col in volume_columns[:2]:
                        if price_col in X.columns and vol_col in X.columns:
                            # Volume-weighted price
                            vwap = (X[price_col] * X[vol_col]).rolling(20).sum() / X[vol_col].rolling(20).sum()
                            enhanced_features[f'{price_col}_{vol_col}_vwap'] = vwap
                            
                            # Price-volume correlation
                            corr = X[price_col].rolling(20).corr(X[vol_col])
                            enhanced_features[f'{price_col}_{vol_col}_corr'] = corr
                            feature_count += 2
            
            # Cross-asset features (if multiple price series)
            if len(price_columns) > 1:
                for i, col1 in enumerate(price_columns[:5]):  # Limit combinations
                    for col2 in price_columns[i+1:6]:
                        if col1 in X.columns and col2 in X.columns:
                            # Price ratio
                            ratio = X[col1] / X[col2]
                            enhanced_features[f'{col1}_{col2}_ratio'] = ratio
                            
                            # Correlation
                            corr = X[col1].rolling(20).corr(X[col2])
                            enhanced_features[f'{col1}_{col2}_corr'] = corr
                            
                            # Spread
                            spread = X[col1] - X[col2]
                            enhanced_features[f'{col1}_{col2}_spread'] = spread
                            feature_count += 3
            
            metadata = {
                'method': 'financial_features',
                'n_new_features': feature_count,
                'price_columns': price_columns,
                'volume_columns': volume_columns,
                'volatility_windows': self.config.volatility_windows,
                'momentum_windows': self.config.momentum_windows
            }
            
            self.logger.info(f"✅ Created {feature_count} financial features")
            
        except Exception as e:
            self.logger.error(f"❌ Financial feature creation failed: {e}")
            enhanced_features = X.copy()
            metadata = {'error': str(e)}
        
        return enhanced_features, metadata


class NeuralFeatureSynthesizer:
    """Neural network-based feature synthesis."""
    
    def __init__(self, config: AutoFeatureConfig):
        self.config = config
        self.logger = system_logger.getChild('NeuralFeatureSynthesizer')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def engineer_features(
        self, 
        X: pd.DataFrame, 
        y: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Synthesize features using neural networks."""
        
        self.logger.info("🧠 Synthesizing features using neural networks")
        
        try:
            # Prepare data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.fillna(0))
            
            # Create neural feature synthesizer
            class FeatureSynthesizer(nn.Module):
                def __init__(self, input_dim, hidden_dims, output_dim):
                    super().__init__()
                    layers = []
                    prev_dim = input_dim
                    
                    for hidden_dim in hidden_dims:
                        layers.append(nn.Linear(prev_dim, hidden_dim))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(0.2))
                        prev_dim = hidden_dim
                    
                    layers.append(nn.Linear(prev_dim, output_dim))
                    self.network = nn.Sequential(*layers)
                
                def forward(self, x):
                    return self.network(x)
            
            # Initialize model
            input_dim = X_scaled.shape[1]
            output_dim = min(20, input_dim // 2)  # Synthesize fewer features than input
            
            model = FeatureSynthesizer(
                input_dim, 
                self.config.synthesis_hidden_dims, 
                output_dim
            ).to(self.device)
            
            # Training data
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            if y is not None:
                # Supervised synthesis - predict target
                y_tensor = torch.FloatTensor(y).to(self.device)
                if len(y_tensor.shape) == 1:
                    y_tensor = y_tensor.unsqueeze(1)
                
                # Modified model for supervised learning
                class SupervisedSynthesizer(nn.Module):
                    def __init__(self, input_dim, hidden_dims, feature_dim, target_dim):
                        super().__init__()
                        # Feature extraction layers
                        feature_layers = []
                        prev_dim = input_dim
                        
                        for hidden_dim in hidden_dims:
                            feature_layers.append(nn.Linear(prev_dim, hidden_dim))
                            feature_layers.append(nn.ReLU())
                            feature_layers.append(nn.Dropout(0.2))
                            prev_dim = hidden_dim
                        
                        feature_layers.append(nn.Linear(prev_dim, feature_dim))
                        self.feature_extractor = nn.Sequential(*feature_layers)
                        
                        # Prediction head
                        self.predictor = nn.Linear(feature_dim, target_dim)
                    
                    def forward(self, x):
                        features = self.feature_extractor(x)
                        predictions = self.predictor(features)
                        return features, predictions
                
                model = SupervisedSynthesizer(
                    input_dim, 
                    self.config.synthesis_hidden_dims, 
                    output_dim,
                    y_tensor.shape[1]
                ).to(self.device)
                
                # Training loop
                optimizer = optim.Adam(model.parameters(), lr=self.config.synthesis_lr)
                criterion = nn.MSELoss()
                
                model.train()
                for epoch in range(self.config.synthesis_epochs):
                    optimizer.zero_grad()
                    features, predictions = model(X_tensor)
                    loss = criterion(predictions, y_tensor)
                    loss.backward()
                    optimizer.step()
                    
                    if epoch % 20 == 0:
                        self.logger.info(f"Synthesis epoch {epoch}, Loss: {loss.item():.6f}")
                
                # Extract synthesized features
                model.eval()
                with torch.no_grad():
                    synthesized_features, _ = model(X_tensor)
                    synthesized_features = synthesized_features.cpu().numpy()
            
            else:
                # Unsupervised synthesis - autoencoder style
                optimizer = optim.Adam(model.parameters(), lr=self.config.synthesis_lr)
                criterion = nn.MSELoss()
                
                # Add decoder for reconstruction
                decoder = nn.Linear(output_dim, input_dim).to(self.device)
                decoder_optimizer = optim.Adam(decoder.parameters(), lr=self.config.synthesis_lr)
                
                model.train()
                decoder.train()
                
                for epoch in range(self.config.synthesis_epochs):
                    optimizer.zero_grad()
                    decoder_optimizer.zero_grad()
                    
                    encoded = model(X_tensor)
                    reconstructed = decoder(encoded)
                    loss = criterion(reconstructed, X_tensor)
                    
                    loss.backward()
                    optimizer.step()
                    decoder_optimizer.step()
                    
                    if epoch % 20 == 0:
                        self.logger.info(f"Synthesis epoch {epoch}, Loss: {loss.item():.6f}")
                
                # Extract synthesized features
                model.eval()
                with torch.no_grad():
                    synthesized_features = model(X_tensor).cpu().numpy()
            
            # Create DataFrame with synthesized features
            feature_names = [f'neural_feature_{i}' for i in range(synthesized_features.shape[1])]
            synthesized_df = pd.DataFrame(
                synthesized_features,
                columns=feature_names,
                index=X.index
            )
            
            # Combine with original features
            enhanced_features = pd.concat([X, synthesized_df], axis=1)
            
            metadata = {
                'method': 'neural_synthesis',
                'n_synthesized_features': synthesized_features.shape[1],
                'synthesis_epochs': self.config.synthesis_epochs,
                'hidden_dims': self.config.synthesis_hidden_dims,
                'supervised': y is not None
            }
            
            self.logger.info(f"✅ Synthesized {synthesized_features.shape[1]} neural features")
            
        except Exception as e:
            self.logger.error(f"❌ Neural feature synthesis failed: {e}")
            enhanced_features = X.copy()
            metadata = {'error': str(e)}
        
        return enhanced_features, metadata


class AutomatedFeatureEngineer:
    """Main class for automated feature engineering."""
    
    def __init__(self, config: AutoFeatureConfig = None):
        self.config = config or AutoFeatureConfig()
        self.logger = system_logger.getChild('AutomatedFeatureEngineer')
        
        # Initialize engineers
        self.engineers = {
            FeatureEngineeringMethod.GENETIC_PROGRAMMING: GeneticFeatureEngineer(self.config),
            FeatureEngineeringMethod.POLYNOMIAL_FEATURES: PolynomialFeatureEngineer(self.config),
            FeatureEngineeringMethod.TIME_SERIES_FEATURES: TimeSeriesFeatureEngineer(self.config),
            FeatureEngineeringMethod.DOMAIN_SPECIFIC_FEATURES: FinancialFeatureEngineer(self.config),
            FeatureEngineeringMethod.NEURAL_FEATURE_SYNTHESIS: NeuralFeatureSynthesizer(self.config)
        }
        
        self.feature_history = []
        self.performance_history = []
    
    def engineer_all_features(
        self,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        methods: Optional[List[FeatureEngineeringMethod]] = None,
        price_columns: Optional[List[str]] = None,
        volume_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply all feature engineering methods."""
        
        self.logger.info("🚀 Starting automated feature engineering pipeline")
        
        if methods is None:
            methods = [
                FeatureEngineeringMethod.TIME_SERIES_FEATURES,
                FeatureEngineeringMethod.DOMAIN_SPECIFIC_FEATURES,
                FeatureEngineeringMethod.POLYNOMIAL_FEATURES,
                FeatureEngineeringMethod.NEURAL_FEATURE_SYNTHESIS
            ]
            # Add genetic programming if available
            if GPLEARN_AVAILABLE:
                methods.append(FeatureEngineeringMethod.GENETIC_PROGRAMMING)
        
        current_features = X.copy()
        all_metadata = {}
        total_new_features = 0
        
        for method in methods:
            try:
                self.logger.info(f"🔧 Applying {method.value}")
                
                if method == FeatureEngineeringMethod.DOMAIN_SPECIFIC_FEATURES:
                    enhanced_features, metadata = self.engineers[method].engineer_features(
                        current_features, price_columns, volume_columns
                    )
                else:
                    enhanced_features, metadata = self.engineers[method].engineer_features(
                        current_features, y
                    )
                
                # Track new features
                new_feature_count = enhanced_features.shape[1] - current_features.shape[1]
                total_new_features += new_feature_count
                
                current_features = enhanced_features
                all_metadata[method.value] = metadata
                
                self.logger.info(f"✅ {method.value} completed: +{new_feature_count} features")
                
                # Prevent feature explosion
                if current_features.shape[1] > self.config.max_features:
                    self.logger.warning(f"Feature count ({current_features.shape[1]}) exceeds limit")
                    current_features = self._select_top_features(current_features, y)
                
            except Exception as e:
                self.logger.error(f"❌ {method.value} failed: {e}")
                all_metadata[method.value] = {'error': str(e)}
        
        # Final feature selection and cleanup
        final_features = self._clean_and_select_features(current_features, y)
        
        # Calculate overall metadata
        overall_metadata = {
            'methods_applied': [method.value for method in methods],
            'total_new_features': total_new_features,
            'final_feature_count': final_features.shape[1],
            'original_feature_count': X.shape[1],
            'method_details': all_metadata,
            'feature_reduction_applied': final_features.shape[1] < current_features.shape[1]
        }
        
        self.logger.info(f"🎯 Feature engineering complete: {X.shape[1]} → {final_features.shape[1]} features")
        
        return final_features, overall_metadata
    
    def _select_top_features(self, X: pd.DataFrame, y: Optional[np.ndarray]) -> pd.DataFrame:
        """Select top features to prevent explosion."""
        
        if y is None or len(X) < 100:
            # Random selection if no target or small dataset
            selected_cols = X.columns[:self.config.max_features]
            return X[selected_cols]
        
        try:
            # Use mutual information for feature selection
            selector = SelectKBest(mutual_info_classif, k=self.config.max_features)
            X_selected = selector.fit_transform(X.fillna(0), y)
            selected_columns = X.columns[selector.get_support()]
            
            return pd.DataFrame(X_selected, columns=selected_columns, index=X.index)
        
        except Exception as e:
            self.logger.warning(f"Feature selection failed: {e}")
            return X.iloc[:, :self.config.max_features]
    
    def _clean_and_select_features(self, X: pd.DataFrame, y: Optional[np.ndarray]) -> pd.DataFrame:
        """Clean and select final features."""
        
        # Remove features with too many NaNs
        nan_threshold = 0.5
        valid_features = X.columns[X.isnull().mean() < nan_threshold]
        X_clean = X[valid_features].copy()
        
        # Remove constant features
        constant_features = X_clean.columns[X_clean.nunique() <= 1]
        X_clean = X_clean.drop(columns=constant_features)
        
        # Remove highly correlated features
        if len(X_clean.columns) > 1:
            corr_matrix = X_clean.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            high_corr_features = [column for column in upper_triangle.columns 
                                if any(upper_triangle[column] > self.config.correlation_threshold)]
            
            X_clean = X_clean.drop(columns=high_corr_features)
        
        # Final feature selection if still too many
        if X_clean.shape[1] > self.config.max_features and y is not None:
            X_clean = self._select_top_features(X_clean, y)
        
        # Fill remaining NaNs
        X_clean = X_clean.fillna(X_clean.mean())
        
        return X_clean
    
    def evaluate_feature_set(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """Evaluate feature set performance."""
        
        try:
            # Determine if classification or regression
            unique_targets = len(np.unique(y))
            is_classification = unique_targets < min(len(y) * 0.1, 50)
            
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                scoring = 'accuracy'
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                scoring = 'r2'
            
            # Cross-validation
            scores = cross_val_score(
                model, X.fillna(0), y, 
                cv=cv_folds, scoring=scoring, n_jobs=self.config.n_jobs
            )
            
            return {
                'mean_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)),
                'min_score': float(np.min(scores)),
                'max_score': float(np.max(scores)),
                'scoring_method': scoring,
                'cv_folds': cv_folds
            }
        
        except Exception as e:
            self.logger.error(f"Feature evaluation failed: {e}")
            return {'error': str(e)}


# Example usage and integration
if __name__ == "__main__":
    # Demo with sample financial data
    np.random.seed(42)
    
    # Create sample market data
    n_samples = 1000
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Simulate price data with trends and volatility
    price = 100 + np.cumsum(np.random.randn(n_samples) * 0.02)
    volume = np.random.lognormal(10, 0.5, n_samples)
    
    # Create additional features
    data = {
        'price': price,
        'volume': volume,
        'high': price * (1 + np.abs(np.random.randn(n_samples) * 0.01)),
        'low': price * (1 - np.abs(np.random.randn(n_samples) * 0.01)),
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples)
    }
    
    market_data = pd.DataFrame(data, index=dates)
    
    # Create target (future returns)
    target = (market_data['price'].shift(-5) / market_data['price'] - 1).fillna(0)
    target_binary = (target > target.median()).astype(int)
    
    # Initialize automated feature engineer
    config = AutoFeatureConfig(
        poly_degree=2,
        rolling_windows=[5, 10, 20],
        lag_features=[1, 2, 3, 5],
        synthesis_epochs=50,
        max_features=500,
        verbose=True
    )
    
    engineer = AutomatedFeatureEngineer(config)
    
    # Apply automated feature engineering
    enhanced_features, metadata = engineer.engineer_all_features(
        market_data.iloc[:-5],  # Remove last 5 rows (target calculation)
        target_binary.iloc[:-5].values,
        price_columns=['price', 'high', 'low'],
        volume_columns=['volume']
    )
    
    print("🎯 Automated Feature Engineering Results:")
    print(f"Original features: {market_data.shape[1]}")
    print(f"Enhanced features: {enhanced_features.shape[1]}")
    print(f"Total new features: {metadata['total_new_features']}")
    
    # Evaluate feature set
    evaluation = engineer.evaluate_feature_set(
        enhanced_features, target_binary.iloc[:-5].values
    )
    print(f"Feature set performance: {evaluation}")
    
    print("\nMethod details:")
    for method, details in metadata['method_details'].items():
        if 'error' not in details:
            print(f"- {method}: Success")
        else:
            print(f"- {method}: {details['error']}")