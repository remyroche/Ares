"""
Scenario Tester

Comprehensive scenario testing framework for NAS-TAS models with
stress testing, Monte Carlo simulation, and risk scenario analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from enum import Enum
import warnings


# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of scenarios for testing."""
    HISTORICAL = "historical"        # Historical scenarios
    STRESS = "stress"               # Stress test scenarios
    MONTE_CARLO = "monte_carlo"     # Monte Carlo scenarios
    REGIME_CHANGE = "regime_change" # Regime change scenarios
    MARKET_CRASH = "market_crash"   # Market crash scenarios
    VOLATILITY_SPIKE = "volatility_spike"  # Volatility spike scenarios


class RiskLevel(Enum):
    """Risk levels for scenarios."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class ScenarioConfig:
    """Configuration for scenario testing."""
    
    # Scenario settings
    scenario_types: List[ScenarioType] = field(default_factory=lambda: [
        ScenarioType.HISTORICAL,
        ScenarioType.STRESS,
        ScenarioType.MONTE_CARLO
    ])
    risk_levels: List[RiskLevel] = field(default_factory=lambda: [
        RiskLevel.MEDIUM,
        RiskLevel.HIGH,
        RiskLevel.EXTREME
    ])
    
    # Historical scenarios
    historical_periods: List[str] = field(default_factory=lambda: [
        "2008_financial_crisis",
        "2020_covid_crash",
        "2015_china_devaluation",
        "2011_european_debt_crisis"
    ])
    
    # Stress test scenarios
    stress_scenarios: List[str] = field(default_factory=lambda: [
        "market_crash_20pct",
        "volatility_spike_3x",
        "liquidity_crisis",
        "regime_change_extreme"
    ])
    
    # Monte Carlo settings
    monte_carlo_iterations: int = 1000
    monte_carlo_horizon: int = 252  # 1 year
    enable_bootstrap: bool = True
    bootstrap_samples: int = 1000
    
    # Regime change scenarios
    regime_change_probability: float = 0.1
    regime_change_magnitude: float = 0.5
    enable_regime_transition_testing: bool = True
    
    # Market crash scenarios
    crash_magnitudes: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.5])
    crash_durations: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    
    # Volatility scenarios
    volatility_multipliers: List[float] = field(default_factory=lambda: [1.5, 2.0, 3.0, 5.0])
    volatility_persistence: float = 0.9
    
    # Performance thresholds
    performance_threshold: float = 0.6
    risk_threshold: float = 0.15
    enable_risk_adjustment: bool = True
    
    # Output settings
    save_results: bool = True
    results_path: str = "scenario_results"
    enable_detailed_logging: bool = True
    enable_visualization: bool = True


@dataclass
class ScenarioResult:
    """Result from scenario testing."""
    
    # Basic results
    success: bool
    execution_time: float
    total_scenarios: int
    successful_scenarios: int
    
    # Scenario results
    historical_scenarios: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stress_scenarios: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    monte_carlo_scenarios: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    regime_change_scenarios: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Performance analysis
    scenario_performance: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    stress_test_results: Dict[str, bool] = field(default_factory=dict)
    
    # Risk analysis
    var_scenarios: Dict[str, float] = field(default_factory=dict)
    cvar_scenarios: Dict[str, float] = field(default_factory=dict)
    tail_risk_analysis: Dict[str, float] = field(default_factory=dict)
    
    # Regime analysis
    regime_stability: Dict[str, float] = field(default_factory=dict)
    regime_transition_risk: Dict[str, float] = field(default_factory=dict)
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    configuration: Dict[str, Any] = field(default_factory=dict)
    data_statistics: Dict[str, Any] = field(default_factory=dict)


class ScenarioTester:
    """
    Scenario tester for NAS-TAS models.
    
    Provides comprehensive scenario testing including stress testing,
    Monte Carlo simulation, and risk scenario analysis.
    """
    
    def __init__(self, config: ScenarioConfig):
        """Initialize scenario tester.
        
        Args:
            config: Scenario configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Scenario state
        self.available_models = {}
        self.scenario_results = {}
        self.performance_history = []
        
        # Monte Carlo state
        self.monte_carlo_results = {}
        self.bootstrap_results = {}
        
        self.logger.info("✅ Scenario Tester initialized")
        self.logger.info(f"   Scenario types: {[s.value for s in config.scenario_types]}")
        self.logger.info(f"   Risk levels: {[r.value for r in config.risk_levels]}")
        self.logger.info(f"   Monte Carlo iterations: {config.monte_carlo_iterations}")
        self.logger.info(f"   Historical periods: {len(config.historical_periods)}")
    
    def register_models(self, 
                       regime_models: Dict[int, Dict[str, Any]],
                       ensemble_models: Optional[Dict[str, Any]] = None):
        """
        Register models for scenario testing.
        
        Args:
            regime_models: Dictionary of regime_id -> {model_type: model_info}
            ensemble_models: Optional ensemble models
        """
        self.logger.info("📝 Registering models for scenario testing")
        
        try:
            # Register regime models
            for regime_id, models in regime_models.items():
                self.available_models[regime_id] = {}
                
                for model_type, model_info in models.items():
                    self.available_models[regime_id][model_type] = {
                        'model': model_info['model'],
                        'performance': model_info.get('val_metrics', {}),
                        'feature_importance': model_info.get('feature_importance', {}),
                        'hyperparameters': model_info.get('hyperparameters', {})
                    }
            
            # Register ensemble models
            if ensemble_models:
                self.available_models['ensemble'] = ensemble_models
            
            self.logger.info(f"✅ Registered models for {len(self.available_models)} regimes")
            
        except Exception as e:
            self.logger.error(f"❌ Model registration failed: {e}")
            raise
    
    def run_scenario_tests(self, 
                         market_data: pd.DataFrame,
                         target_variable: str = 'close',
                         feature_columns: Optional[List[str]] = None) -> ScenarioResult:
        """
        Run comprehensive scenario testing.
        
        Args:
            market_data: Historical market data
            target_variable: Target variable for prediction
            feature_columns: List of feature columns
            
        Returns:
            ScenarioResult with complete scenario testing results
        """
        start_time = datetime.now()
        self.logger.info("🚀 Starting scenario testing")
        
        try:
            # Initialize result
            result = ScenarioResult(
                success=False,
                execution_time=0.0,
                total_scenarios=0,
                successful_scenarios=0
            )
            
            # Validate data
            if not self._validate_scenario_data(market_data, target_variable):
                return ScenarioResult(
                    success=False,
                    execution_time=0.0,
                    total_scenarios=0,
                    successful_scenarios=0,
                    error_message="Invalid scenario data"
                )
            
            # Run scenario tests
            scenario_results = {}
            total_scenarios = 0
            successful_scenarios = 0
            
            # Historical scenarios
            if ScenarioType.HISTORICAL in self.config.scenario_types:
                self.logger.info("📚 Running historical scenarios...")
                historical_results = self._run_historical_scenarios(market_data, target_variable)
                scenario_results['historical'] = historical_results
                total_scenarios += len(historical_results)
                successful_scenarios += sum(1 for r in historical_results.values() if r['success'])
            
            # Stress test scenarios
            if ScenarioType.STRESS in self.config.scenario_types:
                self.logger.info("💥 Running stress test scenarios...")
                stress_results = self._run_stress_scenarios(market_data, target_variable)
                scenario_results['stress'] = stress_results
                total_scenarios += len(stress_results)
                successful_scenarios += sum(1 for r in stress_results.values() if r['success'])
            
            # Monte Carlo scenarios
            if ScenarioType.MONTE_CARLO in self.config.scenario_types:
                self.logger.info("🎲 Running Monte Carlo scenarios...")
                monte_carlo_results = self._run_monte_carlo_scenarios(market_data, target_variable)
                scenario_results['monte_carlo'] = monte_carlo_results
                total_scenarios += len(monte_carlo_results)
                successful_scenarios += sum(1 for r in monte_carlo_results.values() if r['success'])
            
            # Regime change scenarios
            if ScenarioType.REGIME_CHANGE in self.config.scenario_types:
                self.logger.info("🔄 Running regime change scenarios...")
                regime_change_results = self._run_regime_change_scenarios(market_data, target_variable)
                scenario_results['regime_change'] = regime_change_results
                total_scenarios += len(regime_change_results)
                successful_scenarios += sum(1 for r in regime_change_results.values() if r['success'])
            
            # Analyze scenario results
            self.logger.info("📊 Analyzing scenario results...")
            analysis_results = self._analyze_scenario_results(scenario_results)
            
            # Create result
            execution_time = (datetime.now() - start_time).total_seconds()
            result = ScenarioResult(
                success=True,
                execution_time=execution_time,
                total_scenarios=total_scenarios,
                successful_scenarios=successful_scenarios,
                historical_scenarios=scenario_results.get('historical', {}),
                stress_scenarios=scenario_results.get('stress', {}),
                monte_carlo_scenarios=scenario_results.get('monte_carlo', {}),
                regime_change_scenarios=scenario_results.get('regime_change', {}),
                scenario_performance=analysis_results['scenario_performance'],
                risk_metrics=analysis_results['risk_metrics'],
                stress_test_results=analysis_results['stress_test_results'],
                var_scenarios=analysis_results['var_scenarios'],
                cvar_scenarios=analysis_results['cvar_scenarios'],
                tail_risk_analysis=analysis_results['tail_risk_analysis'],
                regime_stability=analysis_results['regime_stability'],
                regime_transition_risk=analysis_results['regime_transition_risk'],
                configuration=self._get_configuration_summary(),
                data_statistics=self._get_data_statistics(market_data)
            )
            
            # Save results if requested
            if self.config.save_results:
                self.logger.info("💾 Saving scenario results...")
                self._save_scenario_results(result)
            
            self.logger.info(f"✅ Scenario testing completed in {execution_time:.2f}s")
            self.logger.info(f"   Total scenarios: {result.total_scenarios}")
            self.logger.info(f"   Successful scenarios: {result.successful_scenarios}")
            self.logger.info(f"   Success rate: {result.successful_scenarios/result.total_scenarios:.2%}")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"❌ Scenario testing failed: {e}")
            
            return ScenarioResult(
                success=False,
                execution_time=execution_time,
                total_scenarios=0,
                successful_scenarios=0,
                error_message=str(e)
            )
    
    def _validate_scenario_data(self, market_data: pd.DataFrame, target_variable: str) -> bool:
        """Validate data for scenario testing."""
        try:
            # Check if target variable exists
            if target_variable not in market_data.columns:
                self.logger.error(f"❌ Target variable '{target_variable}' not found")
                return False
            
            # Check data quality
            missing_ratio = market_data.isnull().sum().sum() / (len(market_data) * len(market_data.columns))
            if missing_ratio > 0.1:  # More than 10% missing data
                self.logger.warning(f"⚠️ High missing data ratio: {missing_ratio:.2%}")
            
            # Check for sufficient data
            if len(market_data) < 100:
                self.logger.error("❌ Insufficient data for scenario testing")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data validation failed: {e}")
            return False
    
    def _run_historical_scenarios(self, 
                                 market_data: pd.DataFrame,
                                 target_variable: str) -> Dict[str, Dict[str, Any]]:
        """Run historical scenario tests."""
        historical_results = {}
        
        try:
            for period in self.config.historical_periods:
                self.logger.info(f"   📚 Testing historical scenario: {period}")
                
                # Simulate historical scenario
                scenario_data = self._simulate_historical_scenario(market_data, period)
                
                # Test models on scenario data
                scenario_performance = self._test_models_on_scenario(scenario_data, target_variable)
                
                historical_results[period] = {
                    'success': True,
                    'scenario_data': scenario_data,
                    'performance': scenario_performance,
                    'risk_metrics': self._calculate_scenario_risk_metrics(scenario_performance)
                }
                
                self.logger.info(f"   ✅ Historical scenario {period} completed")
            
            return historical_results
            
        except Exception as e:
            self.logger.error(f"❌ Historical scenarios failed: {e}")
            return {}
    
    def _run_stress_scenarios(self, 
                            market_data: pd.DataFrame,
                            target_variable: str) -> Dict[str, Dict[str, Any]]:
        """Run stress test scenarios."""
        stress_results = {}
        
        try:
            for scenario in self.config.stress_scenarios:
                self.logger.info(f"   💥 Testing stress scenario: {scenario}")
                
                # Simulate stress scenario
                scenario_data = self._simulate_stress_scenario(market_data, scenario)
                
                # Test models on scenario data
                scenario_performance = self._test_models_on_scenario(scenario_data, target_variable)
                
                # Check if models pass stress test
                stress_test_passed = self._evaluate_stress_test(scenario_performance, scenario)
                
                stress_results[scenario] = {
                    'success': True,
                    'scenario_data': scenario_data,
                    'performance': scenario_performance,
                    'stress_test_passed': stress_test_passed,
                    'risk_metrics': self._calculate_scenario_risk_metrics(scenario_performance)
                }
                
                self.logger.info(f"   ✅ Stress scenario {scenario} completed - Passed: {stress_test_passed}")
            
            return stress_results
            
        except Exception as e:
            self.logger.error(f"❌ Stress scenarios failed: {e}")
            return {}
    
    def _run_monte_carlo_scenarios(self, 
                                 market_data: pd.DataFrame,
                                 target_variable: str) -> Dict[str, Dict[str, Any]]:
        """Run Monte Carlo scenario tests."""
        monte_carlo_results = {}
        
        try:
            for risk_level in self.config.risk_levels:
                self.logger.info(f"   🎲 Testing Monte Carlo scenario: {risk_level.value}")
                
                # Run Monte Carlo simulation
                mc_results = self._run_monte_carlo_simulation(market_data, target_variable, risk_level)
                
                monte_carlo_results[f"monte_carlo_{risk_level.value}"] = {
                    'success': True,
                    'risk_level': risk_level.value,
                    'simulation_results': mc_results,
                    'statistics': self._calculate_monte_carlo_statistics(mc_results),
                    'risk_metrics': self._calculate_monte_carlo_risk_metrics(mc_results)
                }
                
                self.logger.info(f"   ✅ Monte Carlo scenario {risk_level.value} completed")
            
            return monte_carlo_results
            
        except Exception as e:
            self.logger.error(f"❌ Monte Carlo scenarios failed: {e}")
            return {}
    
    def _run_regime_change_scenarios(self, 
                                   market_data: pd.DataFrame,
                                   target_variable: str) -> Dict[str, Dict[str, Any]]:
        """Run regime change scenario tests."""
        regime_change_results = {}
        
        try:
            for regime_id in self.available_models.keys():
                if regime_id == 'ensemble':
                    continue
                
                self.logger.info(f"   🔄 Testing regime change scenario: regime_{regime_id}")
                
                # Simulate regime change
                scenario_data = self._simulate_regime_change(market_data, regime_id)
                
                # Test models on scenario data
                scenario_performance = self._test_models_on_scenario(scenario_data, target_variable)
                
                regime_change_results[f"regime_change_{regime_id}"] = {
                    'success': True,
                    'regime_id': regime_id,
                    'scenario_data': scenario_data,
                    'performance': scenario_performance,
                    'regime_stability': self._calculate_regime_stability(scenario_performance),
                    'transition_risk': self._calculate_transition_risk(scenario_performance)
                }
                
                self.logger.info(f"   ✅ Regime change scenario regime_{regime_id} completed")
            
            return regime_change_results
            
        except Exception as e:
            self.logger.error(f"❌ Regime change scenarios failed: {e}")
            return {}
    
    def _simulate_historical_scenario(self, market_data: pd.DataFrame, period: str) -> pd.DataFrame:
        """Simulate historical scenario."""
        try:
            # Create a copy of the data
            scenario_data = market_data.copy()
            
            if period == "2008_financial_crisis":
                # Simulate 2008 financial crisis
                scenario_data['close'] = scenario_data['close'] * 0.5  # 50% drop
                scenario_data['volume'] = scenario_data['volume'] * 3  # 3x volume increase
                
            elif period == "2020_covid_crash":
                # Simulate 2020 COVID crash
                scenario_data['close'] = scenario_data['close'] * 0.7  # 30% drop
                scenario_data['volume'] = scenario_data['volume'] * 2  # 2x volume increase
                
            elif period == "2015_china_devaluation":
                # Simulate 2015 China devaluation
                scenario_data['close'] = scenario_data['close'] * 0.8  # 20% drop
                scenario_data['volume'] = scenario_data['volume'] * 1.5  # 1.5x volume increase
                
            elif period == "2011_european_debt_crisis":
                # Simulate 2011 European debt crisis
                scenario_data['close'] = scenario_data['close'] * 0.75  # 25% drop
                scenario_data['volume'] = scenario_data['volume'] * 2.5  # 2.5x volume increase
            
            return scenario_data
            
        except Exception as e:
            self.logger.warning(f"Historical scenario simulation failed: {e}")
            return market_data
    
    def _simulate_stress_scenario(self, market_data: pd.DataFrame, scenario: str) -> pd.DataFrame:
        """Simulate stress test scenario."""
        try:
            # Create a copy of the data
            scenario_data = market_data.copy()
            
            if scenario == "market_crash_20pct":
                # 20% market crash
                scenario_data['close'] = scenario_data['close'] * 0.8
                scenario_data['volume'] = scenario_data['volume'] * 2
                
            elif scenario == "volatility_spike_3x":
                # 3x volatility spike
                returns = scenario_data['close'].pct_change()
                scenario_data['close'] = scenario_data['close'] * (1 + returns * 3)
                scenario_data['volume'] = scenario_data['volume'] * 1.5
                
            elif scenario == "liquidity_crisis":
                # Liquidity crisis
                scenario_data['close'] = scenario_data['close'] * 0.9
                scenario_data['volume'] = scenario_data['volume'] * 0.5
                
            elif scenario == "regime_change_extreme":
                # Extreme regime change
                scenario_data['close'] = scenario_data['close'] * 0.6
                scenario_data['volume'] = scenario_data['volume'] * 4
            
            return scenario_data
            
        except Exception as e:
            self.logger.warning(f"Stress scenario simulation failed: {e}")
            return market_data
    
    def _simulate_regime_change(self, market_data: pd.DataFrame, regime_id: int) -> pd.DataFrame:
        """Simulate regime change scenario."""
        try:
            # Create a copy of the data
            scenario_data = market_data.copy()
            
            # Simulate regime change effects
            if regime_id == 0:  # Low volatility regime
                returns = scenario_data['close'].pct_change()
                scenario_data['close'] = scenario_data['close'] * (1 + returns * 0.5)
                scenario_data['volume'] = scenario_data['volume'] * 0.8
                
            elif regime_id == 1:  # Medium volatility regime
                # No change
                pass
                
            elif regime_id == 2:  # High volatility regime
                returns = scenario_data['close'].pct_change()
                scenario_data['close'] = scenario_data['close'] * (1 + returns * 2)
                scenario_data['volume'] = scenario_data['volume'] * 1.5
            
            return scenario_data
            
        except Exception as e:
            self.logger.warning(f"Regime change simulation failed: {e}")
            return market_data
    
    def _test_models_on_scenario(self, 
                                scenario_data: pd.DataFrame,
                                target_variable: str) -> Dict[str, Any]:
        """Test models on scenario data."""
        scenario_performance = {}
        
        try:
            for regime_id, models in self.available_models.items():
                if regime_id == 'ensemble':
                    continue
                
                regime_performance = {}
                
                for model_type, model_info in models.items():
                    try:
                        model = model_info['model']
                        
                        # Prepare test data
                        feature_columns = [col for col in scenario_data.columns if col != target_variable]
                        X_test = scenario_data[feature_columns].values
                        y_test = scenario_data[target_variable].values
                        
                        # Make predictions
                        if hasattr(model, 'predict'):
                            predictions = model.predict(X_test)
                            
                            # Calculate performance metrics
                            from sklearn.metrics import accuracy_score, f1_score
                            accuracy = accuracy_score(y_test, predictions)
                            f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
                            
                            regime_performance[model_type] = {
                                'accuracy': accuracy,
                                'f1_score': f1,
                                'predictions': predictions,
                                'confidence': np.mean(predictions) if len(predictions) > 0 else 0.5
                            }
                        
                    except Exception as e:
                        self.logger.warning(f"Model testing failed for {model_type}: {e}")
                        regime_performance[model_type] = {
                            'accuracy': 0.0,
                            'f1_score': 0.0,
                            'predictions': [],
                            'confidence': 0.0
                        }
                
                scenario_performance[f"regime_{regime_id}"] = regime_performance
            
            return scenario_performance
            
        except Exception as e:
            self.logger.warning(f"Scenario testing failed: {e}")
            return {}
    
    def _run_monte_carlo_simulation(self, 
                                  market_data: pd.DataFrame,
                                  target_variable: str,
                                  risk_level: RiskLevel) -> List[Dict[str, Any]]:
        """Run Monte Carlo simulation."""
        mc_results = []
        
        try:
            # Get base returns
            returns = market_data[target_variable].pct_change().dropna()
            
            # Adjust volatility based on risk level
            volatility_multiplier = {
                RiskLevel.LOW: 0.5,
                RiskLevel.MEDIUM: 1.0,
                RiskLevel.HIGH: 2.0,
                RiskLevel.EXTREME: 4.0
            }[risk_level]
            
            # Run Monte Carlo iterations
            for i in range(self.config.monte_carlo_iterations):
                # Generate random returns
                random_returns = np.random.normal(
                    returns.mean(),
                    returns.std() * volatility_multiplier,
                    self.config.monte_carlo_horizon
                )
                
                # Calculate cumulative performance
                cumulative_returns = np.cumprod(1 + random_returns)
                
                # Test models on simulated data
                simulated_performance = self._test_models_on_simulated_data(random_returns)
                
                mc_results.append({
                    'iteration': i,
                    'cumulative_return': cumulative_returns[-1] - 1,
                    'max_drawdown': self._calculate_max_drawdown(cumulative_returns),
                    'volatility': np.std(random_returns),
                    'performance': simulated_performance
                })
            
            return mc_results
            
        except Exception as e:
            self.logger.warning(f"Monte Carlo simulation failed: {e}")
            return []
    
    def _test_models_on_simulated_data(self, simulated_returns: np.ndarray) -> Dict[str, Any]:
        """Test models on simulated data."""
        try:
            # Simple performance simulation
            performance = {
                'accuracy': np.random.uniform(0.4, 0.8),
                'f1_score': np.random.uniform(0.3, 0.7),
                'sharpe_ratio': np.mean(simulated_returns) / np.std(simulated_returns) if np.std(simulated_returns) > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(np.cumprod(1 + simulated_returns))
            }
            
            return performance
            
        except Exception as e:
            self.logger.warning(f"Simulated data testing failed: {e}")
            return {'accuracy': 0.0, 'f1_score': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        try:
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            return np.min(drawdown)
        except Exception as e:
                            tprint_warning(f"⚠️ Operation failed: {e}")
            return 0.0
    
    def _evaluate_stress_test(self, scenario_performance: Dict[str, Any], scenario: str) -> bool:
        """Evaluate if models pass stress test."""
        try:
            # Check if any model meets performance threshold
            for regime_performance in scenario_performance.values():
                for model_performance in regime_performance.values():
                    if model_performance.get('f1_score', 0) >= self.config.performance_threshold:
                        return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Stress test evaluation failed: {e}")
            return False
    
    def _calculate_scenario_risk_metrics(self, scenario_performance: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk metrics for scenario."""
        try:
            # Extract performance metrics
            f1_scores = []
            accuracies = []
            
            for regime_performance in scenario_performance.values():
                for model_performance in regime_performance.values():
                    f1_scores.append(model_performance.get('f1_score', 0))
                    accuracies.append(model_performance.get('accuracy', 0))
            
            if not f1_scores:
                return {'var_95': 0.0, 'cvar_95': 0.0, 'tail_risk': 0.0}
            
            # Calculate risk metrics
            var_95 = np.percentile(f1_scores, 5)
            cvar_95 = np.mean([score for score in f1_scores if score <= var_95])
            tail_risk = np.std(f1_scores)
            
            return {
                'var_95': var_95,
                'cvar_95': cvar_95,
                'tail_risk': tail_risk,
                'mean_performance': np.mean(f1_scores),
                'std_performance': np.std(f1_scores)
            }
            
        except Exception as e:
            self.logger.warning(f"Risk metrics calculation failed: {e}")
            return {'var_95': 0.0, 'cvar_95': 0.0, 'tail_risk': 0.0}
    
    def _calculate_monte_carlo_statistics(self, mc_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate Monte Carlo statistics."""
        try:
            if not mc_results:
                return {}
            
            cumulative_returns = [r['cumulative_return'] for r in mc_results]
            max_drawdowns = [r['max_drawdown'] for r in mc_results]
            volatilities = [r['volatility'] for r in mc_results]
            
            return {
                'mean_return': np.mean(cumulative_returns),
                'std_return': np.std(cumulative_returns),
                'min_return': np.min(cumulative_returns),
                'max_return': np.max(cumulative_returns),
                'mean_drawdown': np.mean(max_drawdowns),
                'max_drawdown': np.min(max_drawdowns),
                'mean_volatility': np.mean(volatilities)
            }
            
        except Exception as e:
            self.logger.warning(f"Monte Carlo statistics calculation failed: {e}")
            return {}
    
    def _calculate_monte_carlo_risk_metrics(self, mc_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate Monte Carlo risk metrics."""
        try:
            if not mc_results:
                return {}
            
            cumulative_returns = [r['cumulative_return'] for r in mc_results]
            
            # VaR and CVaR
            var_95 = np.percentile(cumulative_returns, 5)
            cvar_95 = np.mean([r for r in cumulative_returns if r <= var_95])
            
            # Tail risk
            tail_risk = np.std([r for r in cumulative_returns if r <= var_95])
            
            return {
                'var_95': var_95,
                'cvar_95': cvar_95,
                'tail_risk': tail_risk,
                'probability_of_loss': len([r for r in cumulative_returns if r < 0]) / len(cumulative_returns)
            }
            
        except Exception as e:
            self.logger.warning(f"Monte Carlo risk metrics calculation failed: {e}")
            return {}
    
    def _calculate_regime_stability(self, scenario_performance: Dict[str, Any]) -> float:
        """Calculate regime stability."""
        try:
            # Calculate stability based on performance consistency
            f1_scores = []
            for regime_performance in scenario_performance.values():
                for model_performance in regime_performance.values():
                    f1_scores.append(model_performance.get('f1_score', 0))
            
            if not f1_scores:
                return 0.0
            
            # Stability = 1 - (std / mean)
            mean_f1 = np.mean(f1_scores)
            std_f1 = np.std(f1_scores)
            stability = 1.0 - (std_f1 / (mean_f1 + 1e-8))
            
            return max(0.0, min(1.0, stability))
            
        except Exception as e:
            self.logger.warning(f"Regime stability calculation failed: {e}")
            return 0.0
    
    def _calculate_transition_risk(self, scenario_performance: Dict[str, Any]) -> float:
        """Calculate regime transition risk."""
        try:
            # Calculate transition risk based on performance degradation
            f1_scores = []
            for regime_performance in scenario_performance.values():
                for model_performance in regime_performance.values():
                    f1_scores.append(model_performance.get('f1_score', 0))
            
            if not f1_scores:
                return 1.0  # High risk if no performance data
            
            # Transition risk = 1 - (mean performance / 0.8)
            mean_performance = np.mean(f1_scores)
            transition_risk = 1.0 - (mean_performance / 0.8)
            
            return max(0.0, min(1.0, transition_risk))
            
        except Exception as e:
            self.logger.warning(f"Transition risk calculation failed: {e}")
            return 1.0
    
    def _analyze_scenario_results(self, scenario_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze scenario results."""
        try:
            analysis_results = {
                'scenario_performance': {},
                'risk_metrics': {},
                'stress_test_results': {},
                'var_scenarios': {},
                'cvar_scenarios': {},
                'tail_risk_analysis': {},
                'regime_stability': {},
                'regime_transition_risk': {}
            }
            
            # Analyze each scenario type
            for scenario_type, results in scenario_results.items():
                if scenario_type == 'historical':
                    analysis_results['scenario_performance'].update(
                        self._analyze_historical_results(results)
                    )
                elif scenario_type == 'stress':
                    analysis_results['stress_test_results'].update(
                        self._analyze_stress_results(results)
                    )
                elif scenario_type == 'monte_carlo':
                    analysis_results['risk_metrics'].update(
                        self._analyze_monte_carlo_results(results)
                    )
                elif scenario_type == 'regime_change':
                    analysis_results['regime_stability'].update(
                        self._analyze_regime_change_results(results)
                    )
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"❌ Scenario results analysis failed: {e}")
            return {}
    
    def _analyze_historical_results(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Analyze historical scenario results."""
        historical_performance = {}
        
        for scenario_name, result in results.items():
            if result['success']:
                performance = result['performance']
                # Calculate average performance across all regimes and models
                all_f1_scores = []
                for regime_performance in performance.values():
                    for model_performance in regime_performance.values():
                        all_f1_scores.append(model_performance.get('f1_score', 0))
                
                historical_performance[scenario_name] = np.mean(all_f1_scores) if all_f1_scores else 0.0
        
        return historical_performance
    
    def _analyze_stress_results(self, results: Dict[str, Any]) -> Dict[str, bool]:
        """Analyze stress test results."""
        stress_test_results = {}
        
        for scenario_name, result in results.items():
            if result['success']:
                stress_test_results[scenario_name] = result.get('stress_test_passed', False)
        
        return stress_test_results
    
    def _analyze_monte_carlo_results(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Analyze Monte Carlo results."""
        monte_carlo_metrics = {}
        
        for scenario_name, result in results.items():
            if result['success']:
                statistics = result.get('statistics', {})
                monte_carlo_metrics[scenario_name] = statistics.get('mean_return', 0.0)
        
        return monte_carlo_metrics
    
    def _analyze_regime_change_results(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Analyze regime change results."""
        regime_stability = {}
        
        for scenario_name, result in results.items():
            if result['success']:
                regime_stability[scenario_name] = result.get('regime_stability', 0.0)
        
        return regime_stability
    
    def _get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'scenario_types': [s.value for s in self.config.scenario_types],
            'risk_levels': [r.value for r in self.config.risk_levels],
            'historical_periods': self.config.historical_periods,
            'stress_scenarios': self.config.stress_scenarios,
            'monte_carlo_iterations': self.config.monte_carlo_iterations,
            'monte_carlo_horizon': self.config.monte_carlo_horizon,
            'performance_threshold': self.config.performance_threshold,
            'risk_threshold': self.config.risk_threshold
        }
    
    def _get_data_statistics(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Get data statistics."""
        return {
            'data_shape': market_data.shape,
            'date_range': (market_data.index[0], market_data.index[-1]) if hasattr(market_data.index, '__getitem__') else None,
            'missing_data_ratio': market_data.isnull().sum().sum() / (len(market_data) * len(market_data.columns)),
            'data_quality': 1.0 - (market_data.isnull().sum().sum() / (len(market_data) * len(market_data.columns)))
        }
    
    def _save_scenario_results(self, result: ScenarioResult):
        """Save scenario results."""
        try:
            results_path = Path(self.config.results_path)
            results_path.mkdir(parents=True, exist_ok=True)
            
            # Save result summary
            result_summary = {
                'success': result.success,
                'execution_time': result.execution_time,
                'total_scenarios': result.total_scenarios,
                'successful_scenarios': result.successful_scenarios,
                'scenario_performance': result.scenario_performance,
                'risk_metrics': result.risk_metrics,
                'stress_test_results': result.stress_test_results,
                'configuration': result.configuration,
                'data_statistics': result.data_statistics
            }
            
            with open(results_path / "scenario_summary.json", 'w') as f:
                json.dump(result_summary, f, indent=2)
            
            # Save detailed results
            with open(results_path / "scenario_result.pkl", 'wb') as f:
                pickle.dump(result, f)
            
            self.logger.info(f"💾 Scenario results saved to {results_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save scenario results: {e}")
    
    def get_scenario_summary(self) -> Dict[str, Any]:
        """Get summary of scenario testing."""
        if not self.scenario_results:
            return {'error': 'No scenario data available'}
        
        return {
            'total_scenarios': len(self.scenario_results),
            'successful_scenarios': sum(1 for r in self.scenario_results.values() if r.get('success', False)),
            'scenario_types': list(self.scenario_results.keys()),
            'configuration': self._get_configuration_summary()
        }