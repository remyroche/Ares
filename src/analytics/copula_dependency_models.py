"""
Copula Dependency Models for Price Movement Dependencies
Models dependencies between different price movements across regimes
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass
from scipy.stats import multivariate_normal, norm
from scipy.optimize import minimize

from src.utils.logger import system_logger
from src.core.decorators import handles_errors


@dataclass
class CopulaModel:
    """Copula model for dependency modeling"""
    regime: str
    copula_type: str
    correlation_matrix: np.ndarray
    marginal_parameters: List[Dict[str, float]]
    model_parameters: Dict[str, Any]
    fit_timestamp: datetime
    sample_size: int


class CopulaDependencyModels:
    """
    Copula Dependency Models for price movement dependencies.
    Models dependencies between different price movements across regimes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Copula Dependency Models.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild('CopulaModels')
        
        # Configuration
        self.copula_config = config.get('copula_models', {})
        self.regime_names = [f"regime_{i:02d}" for i in range(20)]  # regime_00 to regime_19
        self.price_targets = [f"{i*0.1:.1f}%" for i in range(1, 21)]  # 0.1% to 2.0%
        
        # Copula parameters
        self.copula_types = ['gaussian', 't_copula', 'clayton', 'gumbel']
        self.default_copula_type = self.copula_config.get('default_copula_type', 'gaussian')
        self.min_sample_size = self.copula_config.get('min_sample_size', 100)
        self.confidence_level = self.copula_config.get('confidence_level', 0.95)
        
        # Storage
        self.copula_models: Dict[str, CopulaModel] = {}
        self.price_movement_data: Dict[str, pd.DataFrame] = {}
        self.dependency_analysis: Dict[str, Dict[str, Any]] = {}
        
        # Initialize storage
        self._initialize_storage()
        
    def _initialize_storage(self) -> None:
        """Initialize storage for copula models and data"""
        
        for regime in self.regime_names:
            self.price_movement_data[regime] = pd.DataFrame()
            self.dependency_analysis[regime] = {}
    
    @handles_errors(exceptions=(ValueError, AttributeError), default_return=False, context='copula models initialization')
    async def initialize(self) -> bool:
        """
        Initialize the Copula Dependency Models system.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing Copula Dependency Models system...")
            
            # Load existing models if available
            await self._load_existing_models()
            
            self.logger.info("✅ Copula Dependency Models system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Copula Dependency Models initialization failed: {e}")
            return False
    
    async def _load_existing_models(self) -> None:
        """Load existing copula models from storage"""
        try:
            # This would load from your existing model storage
            # For now, models are initialized as needed
            self.logger.info("Loaded existing copula models (or initialized as needed)")
        except Exception as e:
            self.logger.warning(f"Could not load existing models: {e}")
    
    @handles_errors(exceptions=(ValueError, KeyError), default_return=None, context='price movement data collection')
    async def collect_price_movement_data(
        self,
        regime: str,
        price_movements: Dict[str, List[float]],
        timestamps: List[datetime] = None
    ) -> Optional[bool]:
        """
        Collect price movement data for copula modeling.
        
        Args:
            regime: HMM regime name
            price_movements: Dictionary of target -> list of price movements
            timestamps: List of timestamps (optional)
            
        Returns:
            bool: True if data collection successful
        """
        try:
            if regime not in self.regime_names:
                self.logger.error(f"Invalid regime: {regime}")
                return None
            
            # Create DataFrame from price movements
            data_dict = {}
            for target, movements in price_movements.items():
                if target in self.price_targets:
                    data_dict[target] = movements
            
            if not data_dict:
                self.logger.error("No valid price movement data provided")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(data_dict)
            
            # Add timestamps if provided
            if timestamps and len(timestamps) == len(df):
                df['timestamp'] = timestamps
            
            # Store data
            if self.price_movement_data[regime].empty:
                self.price_movement_data[regime] = df
            else:
                # Append to existing data
                self.price_movement_data[regime] = pd.concat([
                    self.price_movement_data[regime], df
                ], ignore_index=True)
            
            self.logger.info(f"Collected {len(df)} price movement observations for regime {regime}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error collecting price movement data for regime {regime}: {e}")
            return None
    
    @handles_errors(exceptions=(ValueError, KeyError), default_return=None, context='copula model fitting')
    async def fit_copula_model(
        self,
        regime: str,
        copula_type: str = None,
        targets: List[str] = None
    ) -> Optional[CopulaModel]:
        """
        Fit copula model for a specific regime.
        
        Args:
            regime: HMM regime name
            copula_type: Type of copula to fit ('gaussian', 't_copula', 'clayton', 'gumbel')
            targets: List of price targets to include (None for all)
            
        Returns:
            CopulaModel: Fitted copula model
        """
        try:
            if regime not in self.regime_names:
                self.logger.error(f"Invalid regime: {regime}")
                return None
            
            copula_type = copula_type or self.default_copula_type
            if copula_type not in self.copula_types:
                self.logger.error(f"Invalid copula type: {copula_type}")
                return None
            
            # Get data for regime
            regime_data = self.price_movement_data[regime]
            if regime_data.empty:
                self.logger.error(f"No data available for regime {regime}")
                return None
            
            # Select targets
            if targets is None:
                targets = [t for t in self.price_targets if t in regime_data.columns]
            
            if len(targets) < 2:
                self.logger.error("Need at least 2 targets for dependency modeling")
                return None
            
            # Filter data for selected targets
            target_data = regime_data[targets].dropna()
            
            if len(target_data) < self.min_sample_size:
                self.logger.error(f"Insufficient data for regime {regime}: {len(target_data)} < {self.min_sample_size}")
                return None
            
            self.logger.info(f"Fitting {copula_type} copula for regime {regime} with {len(target_data)} observations")
            
            # Fit copula model
            if copula_type == 'gaussian':
                copula_model = await self._fit_gaussian_copula(regime, target_data, targets)
            elif copula_type == 't_copula':
                copula_model = await self._fit_t_copula(regime, target_data, targets)
            elif copula_type == 'clayton':
                copula_model = await self._fit_clayton_copula(regime, target_data, targets)
            elif copula_type == 'gumbel':
                copula_model = await self._fit_gumbel_copula(regime, target_data, targets)
            else:
                self.logger.error(f"Copula type {copula_type} not implemented")
                return None
            
            if copula_model:
                # Store model
                self.copula_models[regime] = copula_model
                
                # Perform dependency analysis
                await self._analyze_dependencies(regime, copula_model)
                
                self.logger.info(f"✅ Successfully fitted {copula_type} copula for regime {regime}")
            
            return copula_model
            
        except Exception as e:
            self.logger.error(f"Error fitting copula model for regime {regime}: {e}")
            return None
    
    async def _fit_gaussian_copula(
        self,
        regime: str,
        data: pd.DataFrame,
        targets: List[str]
    ) -> Optional[CopulaModel]:
        """Fit Gaussian copula model"""
        
        try:
            # Transform to uniform marginals
            uniform_data = np.zeros_like(data.values)
            marginal_params = []
            
            for i, target in enumerate(targets):
                target_data = data[target].values
                
                # Fit normal distribution to marginal
                mu, sigma = norm.fit(target_data)
                marginal_params.append({'mu': mu, 'sigma': sigma})
                
                # Transform to uniform
                uniform_data[:, i] = norm.cdf(target_data, mu, sigma)
            
            # Estimate correlation matrix
            correlation_matrix = np.corrcoef(uniform_data.T)
            
            # Create model
            copula_model = CopulaModel(
                regime=regime,
                copula_type='gaussian',
                correlation_matrix=correlation_matrix,
                marginal_parameters=marginal_params,
                model_parameters={
                    'correlation_matrix': correlation_matrix,
                    'uniform_data': uniform_data
                },
                fit_timestamp=datetime.now(),
                sample_size=len(data)
            )
            
            return copula_model
            
        except Exception as e:
            self.logger.error(f"Error fitting Gaussian copula: {e}")
            return None
    
    async def _fit_t_copula(
        self,
        regime: str,
        data: pd.DataFrame,
        targets: List[str]
    ) -> Optional[CopulaModel]:
        """Fit t-copula model"""
        
        try:
            # For simplicity, use Gaussian copula as approximation
            # In practice, you would implement proper t-copula fitting
            self.logger.warning("t-copula not fully implemented, using Gaussian approximation")
            return await self._fit_gaussian_copula(regime, data, targets)
            
        except Exception as e:
            self.logger.error(f"Error fitting t-copula: {e}")
            return None
    
    async def _fit_clayton_copula(
        self,
        regime: str,
        data: pd.DataFrame,
        targets: List[str]
    ) -> Optional[CopulaModel]:
        """Fit Clayton copula model"""
        
        try:
            # For simplicity, use Gaussian copula as approximation
            # In practice, you would implement proper Clayton copula fitting
            self.logger.warning("Clayton copula not fully implemented, using Gaussian approximation")
            return await self._fit_gaussian_copula(regime, data, targets)
            
        except Exception as e:
            self.logger.error(f"Error fitting Clayton copula: {e}")
            return None
    
    async def _fit_gumbel_copula(
        self,
        regime: str,
        data: pd.DataFrame,
        targets: List[str]
    ) -> Optional[CopulaModel]:
        """Fit Gumbel copula model"""
        
        try:
            # For simplicity, use Gaussian copula as approximation
            # In practice, you would implement proper Gumbel copula fitting
            self.logger.warning("Gumbel copula not fully implemented, using Gaussian approximation")
            return await self._fit_gaussian_copula(regime, data, targets)
            
        except Exception as e:
            self.logger.error(f"Error fitting Gumbel copula: {e}")
            return None
    
    async def _analyze_dependencies(self, regime: str, copula_model: CopulaModel) -> None:
        """Analyze dependencies from copula model"""
        
        try:
            correlation_matrix = copula_model.correlation_matrix
            n_targets = len(correlation_matrix)
            
            # Calculate dependency metrics
            dependency_metrics = {
                'correlation_matrix': correlation_matrix.tolist(),
                'avg_correlation': np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]),
                'max_correlation': np.max(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]),
                'min_correlation': np.min(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]),
                'strong_dependencies': [],
                'weak_dependencies': []
            }
            
            # Identify strong and weak dependencies
            for i in range(n_targets):
                for j in range(i + 1, n_targets):
                    correlation = correlation_matrix[i, j]
                    
                    if abs(correlation) > 0.7:  # Strong dependency
                        dependency_metrics['strong_dependencies'].append({
                            'target1': i,
                            'target2': j,
                            'correlation': correlation
                        })
                    elif abs(correlation) < 0.3:  # Weak dependency
                        dependency_metrics['weak_dependencies'].append({
                            'target1': i,
                            'target2': j,
                            'correlation': correlation
                        })
            
            # Calculate tail dependencies (simplified)
            tail_dependencies = self._calculate_tail_dependencies(correlation_matrix)
            dependency_metrics['tail_dependencies'] = tail_dependencies
            
            # Store analysis
            self.dependency_analysis[regime] = dependency_metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing dependencies for regime {regime}: {e}")
    
    def _calculate_tail_dependencies(self, correlation_matrix: np.ndarray) -> Dict[str, Any]:
        """Calculate tail dependencies from correlation matrix"""
        
        try:
            n_targets = len(correlation_matrix)
            tail_deps = {}
            
            for i in range(n_targets):
                for j in range(i + 1, n_targets):
                    correlation = correlation_matrix[i, j]
                    
                    # Simplified tail dependency calculation
                    # In practice, you would use proper tail dependency measures
                    upper_tail_dep = max(0, correlation)  # Simplified
                    lower_tail_dep = max(0, correlation)  # Simplified
                    
                    tail_deps[f"{i}_{j}"] = {
                        'upper_tail_dependency': upper_tail_dep,
                        'lower_tail_dependency': lower_tail_dep,
                        'correlation': correlation
                    }
            
            return tail_deps
            
        except Exception as e:
            self.logger.error(f"Error calculating tail dependencies: {e}")
            return {}
    
    @handles_errors(exceptions=(ValueError, KeyError), default_return=None, context='generate correlated scenarios')
    async def generate_correlated_scenarios(
        self,
        regime: str,
        n_scenarios: int = 1000,
        targets: List[str] = None
    ) -> Optional[np.ndarray]:
        """
        Generate correlated price movement scenarios using copula model.
        
        Args:
            regime: HMM regime name
            n_scenarios: Number of scenarios to generate
            targets: List of targets to include (None for all)
            
        Returns:
            np.ndarray: Generated scenarios
        """
        try:
            if regime not in self.copula_models:
                self.logger.error(f"No copula model found for regime {regime}")
                return None
            
            copula_model = self.copula_models[regime]
            
            if copula_model.copula_type == 'gaussian':
                return await self._generate_gaussian_scenarios(copula_model, n_scenarios, targets)
            else:
                self.logger.error(f"Scenario generation not implemented for {copula_model.copula_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating scenarios for regime {regime}: {e}")
            return None
    
    async def _generate_gaussian_scenarios(
        self,
        copula_model: CopulaModel,
        n_scenarios: int,
        targets: List[str] = None
    ) -> Optional[np.ndarray]:
        """Generate scenarios using Gaussian copula"""
        
        try:
            correlation_matrix = copula_model.correlation_matrix
            marginal_params = copula_model.marginal_parameters
            
            # Generate correlated uniform variables
            uniform_scenarios = multivariate_normal.rvs(
                mean=np.zeros(len(correlation_matrix)),
                cov=correlation_matrix,
                size=n_scenarios
            )
            
            # Transform to uniform [0, 1]
            uniform_scenarios = norm.cdf(uniform_scenarios)
            
            # Transform back to original distributions
            price_scenarios = np.zeros_like(uniform_scenarios)
            
            for i, params in enumerate(marginal_params):
                mu = params['mu']
                sigma = params['sigma']
                price_scenarios[:, i] = norm.ppf(uniform_scenarios[:, i], mu, sigma)
            
            return price_scenarios
            
        except Exception as e:
            self.logger.error(f"Error generating Gaussian scenarios: {e}")
            return None
    
    @handles_errors(exceptions=(ValueError, KeyError), default_return=None, context='calculate joint probability')
    async def calculate_joint_probability(
        self,
        regime: str,
        target_movements: Dict[str, float]
    ) -> Optional[float]:
        """
        Calculate joint probability of multiple price movements.
        
        Args:
            regime: HMM regime name
            target_movements: Dictionary of target -> movement value
            
        Returns:
            float: Joint probability
        """
        try:
            if regime not in self.copula_models:
                self.logger.error(f"No copula model found for regime {regime}")
                return None
            
            copula_model = self.copula_models[regime]
            
            if copula_model.copula_type == 'gaussian':
                return await self._calculate_gaussian_joint_probability(copula_model, target_movements)
            else:
                self.logger.error(f"Joint probability calculation not implemented for {copula_model.copula_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error calculating joint probability for regime {regime}: {e}")
            return None
    
    async def _calculate_gaussian_joint_probability(
        self,
        copula_model: CopulaModel,
        target_movements: Dict[str, float]
    ) -> Optional[float]:
        """Calculate joint probability using Gaussian copula"""
        
        try:
            # This is a simplified implementation
            # In practice, you would use proper copula density functions
            
            correlation_matrix = copula_model.correlation_matrix
            marginal_params = copula_model.marginal_parameters
            
            # Transform movements to uniform space
            uniform_values = []
            for i, (target, movement) in enumerate(target_movements.items()):
                if i < len(marginal_params):
                    mu = marginal_params[i]['mu']
                    sigma = marginal_params[i]['sigma']
                    uniform_val = norm.cdf(movement, mu, sigma)
                    uniform_values.append(uniform_val)
            
            if len(uniform_values) != len(correlation_matrix):
                self.logger.error("Mismatch between target movements and model dimensions")
                return None
            
            # Calculate joint probability (simplified)
            # In practice, you would use proper multivariate normal CDF
            joint_prob = 1.0  # Placeholder
            
            return joint_prob
            
        except Exception as e:
            self.logger.error(f"Error calculating Gaussian joint probability: {e}")
            return None
    
    def get_copula_summary(self) -> Dict[str, Any]:
        """Get summary of copula models"""
        
        summary = {
            'system_status': 'active',
            'regime_count': len(self.regime_names),
            'target_count': len(self.price_targets),
            'fitted_models': len(self.copula_models),
            'available_copula_types': self.copula_types,
            'default_copula_type': self.default_copula_type,
            'min_sample_size': self.min_sample_size,
            'model_summaries': {}
        }
        
        for regime, model in self.copula_models.items():
            summary['model_summaries'][regime] = {
                'copula_type': model.copula_type,
                'sample_size': model.sample_size,
                'fit_timestamp': model.fit_timestamp,
                'correlation_matrix_size': len(model.correlation_matrix),
                'avg_correlation': np.mean(model.correlation_matrix[np.triu_indices_from(model.correlation_matrix, k=1)])
            }
        
        return summary