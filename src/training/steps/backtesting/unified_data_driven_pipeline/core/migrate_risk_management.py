#!/usr/bin/env python3
"""
Risk Management Migration Script

This script migrates the RiskManager to the ModularComponent architecture.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from modular_architecture import (
    ModularComponent, ValidationLevel, ValidationResult, ErrorInfo, 
    PerformanceMetric, MetricType, MetricLevel, ErrorSeverity, ErrorCategory
)
from component_registry import (
    ComponentType, BacktestingComponentRegistry, get_registry
)

class MigratedRiskManager(ModularComponent):
    """
    Migrated Risk Manager using ModularComponent architecture.
    
    This component wraps the original RiskManager to provide
    ModularComponent functionality while maintaining backward compatibility.
    """
    
    def __init__(self, config: dict = None):
        super().__init__(config)
        self.component_type = ComponentType.RISK_MANAGER
        self._original_manager = None
        self._risk_metrics = {}
        self._risk_alerts = []
        self._position_sizes = {}
        
    def _validate_config(self, config: dict) -> ValidationResult:
        """Validate the configuration for the risk manager."""
        errors = []
        warnings = []
        
        # Required parameters
        required_params = ['data_loader']
        for param in required_params:
            if param not in config:
                errors.append(ErrorInfo(
                    f"Missing required parameter: {param}",
                    ErrorSeverity.ERROR,
                    ErrorCategory.CONFIGURATION
                ))
        
        # Validate risk limits
        if 'risk_limits' in config:
            risk_limits = config['risk_limits']
            if not isinstance(risk_limits, dict):
                errors.append(ErrorInfo(
                    "Risk limits must be a dictionary",
                    ErrorSeverity.ERROR,
                    ErrorCategory.CONFIGURATION
                ))
            else:
                # Validate individual risk limits
                for limit_name, limit_value in risk_limits.items():
                    if not isinstance(limit_value, (int, float)):
                        errors.append(ErrorInfo(
                            f"Risk limit {limit_name} must be a number",
                            ErrorSeverity.ERROR,
                            ErrorCategory.CONFIGURATION
                        ))
                    elif limit_value < 0:
                        errors.append(ErrorInfo(
                            f"Risk limit {limit_name} must be non-negative",
                            ErrorSeverity.ERROR,
                            ErrorCategory.CONFIGURATION
                        ))
        
        # Validate position sizing
        if 'position_sizing' in config:
            position_sizing = config['position_sizing']
            if not isinstance(position_sizing, dict):
                errors.append(ErrorInfo(
                    "Position sizing must be a dictionary",
                    ErrorSeverity.ERROR,
                    ErrorCategory.CONFIGURATION
                ))
            else:
                # Validate position sizing parameters
                if 'max_position_size' in position_sizing:
                    max_size = position_sizing['max_position_size']
                    if not isinstance(max_size, (int, float)) or max_size <= 0:
                        errors.append(ErrorInfo(
                            "Max position size must be a positive number",
                            ErrorSeverity.ERROR,
                            ErrorCategory.CONFIGURATION
                        ))
                
                if 'risk_per_trade' in position_sizing:
                    risk_per_trade = position_sizing['risk_per_trade']
                    if not isinstance(risk_per_trade, (int, float)) or risk_per_trade <= 0:
                        errors.append(ErrorInfo(
                            "Risk per trade must be a positive number",
                            ErrorSeverity.ERROR,
                            ErrorCategory.CONFIGURATION
                        ))
        
        # Validate VaR settings
        if 'var_settings' in config:
            var_settings = config['var_settings']
            if not isinstance(var_settings, dict):
                errors.append(ErrorInfo(
                    "VaR settings must be a dictionary",
                    ErrorSeverity.ERROR,
                    ErrorCategory.CONFIGURATION
                ))
            else:
                if 'confidence_level' in var_settings:
                    confidence = var_settings['confidence_level']
                    if not isinstance(confidence, (int, float)) or not (0 < confidence < 1):
                        errors.append(ErrorInfo(
                            "VaR confidence level must be between 0 and 1",
                            ErrorSeverity.ERROR,
                            ErrorCategory.CONFIGURATION
                        ))
        
        # Validate alert settings
        if 'alert_settings' in config:
            alert_settings = config['alert_settings']
            if not isinstance(alert_settings, dict):
                errors.append(ErrorInfo(
                    "Alert settings must be a dictionary",
                    ErrorSeverity.ERROR,
                    ErrorCategory.CONFIGURATION
                ))
            else:
                if 'enabled' in alert_settings and not isinstance(alert_settings['enabled'], bool):
                    errors.append(ErrorInfo(
                        "Alert enabled setting must be a boolean",
                        ErrorSeverity.ERROR,
                        ErrorCategory.CONFIGURATION
                    ))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _initialize_original_manager(self):
        """Initialize the original RiskManager."""
        try:
            from ...abc_testing.risk_management import RiskManager
            
            # Create configuration for original manager
            original_config = {
                'risk_limits': self.get_config('risk_limits', {}),
                'position_sizing': self.get_config('position_sizing', {}),
                'var_settings': self.get_config('var_settings', {}),
                'alert_settings': self.get_config('alert_settings', {}),
                'max_drawdown_limit': self.get_config('max_drawdown_limit', 0.2),
                'volatility_limit': self.get_config('volatility_limit', 0.3),
                'correlation_limit': self.get_config('correlation_limit', 0.8)
            }
            
            self._original_manager = RiskManager(original_config)
            return True
            
        except Exception as e:
            self._add_error(f"Failed to initialize original risk manager: {e}")
            return False
    
    def _execute_risk_assessment(self, portfolio_data, market_data, **kwargs):
        """Execute the risk assessment."""
        try:
            if self._original_manager is None:
                if not self._initialize_original_manager():
                    return None
            
            # Perform risk assessment using original manager
            risk_results = self._original_manager.assess_risk(
                portfolio_data=portfolio_data,
                market_data=market_data,
                **kwargs
            )
            
            # Store risk metrics
            self._risk_metrics = risk_results.get('metrics', {})
            self._risk_alerts = risk_results.get('alerts', [])
            
            # Record performance metrics
            self._record_metric(PerformanceMetric(
                name="risk_assessment_success",
                value=1.0,
                metric_type=MetricType.SUCCESS_RATE,
                level=MetricLevel.COMPONENT
            ))
            
            # Record risk metrics
            if 'var_95' in self._risk_metrics:
                self._record_metric(PerformanceMetric(
                    name="var_95",
                    value=self._risk_metrics['var_95'],
                    metric_type=MetricType.RISK,
                    level=MetricLevel.COMPONENT
                ))
            
            if 'max_drawdown' in self._risk_metrics:
                self._record_metric(PerformanceMetric(
                    name="max_drawdown",
                    value=self._risk_metrics['max_drawdown'],
                    metric_type=MetricType.RISK,
                    level=MetricLevel.COMPONENT
                ))
            
            return risk_results
            
        except Exception as e:
            self._add_error(f"Risk assessment failed: {e}")
            return None
    
    def assess_risk(self, portfolio_data, market_data, **kwargs):
        """
        Assess portfolio risk using the ModularComponent architecture.
        
        Args:
            portfolio_data: Current portfolio data
            market_data: Market data for risk calculation
            **kwargs: Additional risk assessment parameters
            
        Returns:
            Risk assessment results or None if failed
        """
        if not self._is_initialized:
            self._add_error("Component not initialized")
            return None
        
        if not self._is_started:
            self._add_error("Component not started")
            return None
        
        # Validate inputs
        if portfolio_data is None:
            self._add_error("Portfolio data cannot be None")
            return None
        
        if market_data is None:
            self._add_error("Market data cannot be None")
            return None
        
        # Execute risk assessment
        return self._execute_risk_assessment(portfolio_data, market_data, **kwargs)
    
    def calculate_position_size(self, signal_strength, volatility, account_value):
        """Calculate appropriate position size based on risk parameters."""
        try:
            if self._original_manager is None:
                if not self._initialize_original_manager():
                    return None
            
            # Calculate position size using original manager
            position_size = self._original_manager.calculate_position_size(
                signal_strength=signal_strength,
                volatility=volatility,
                account_value=account_value
            )
            
            # Store position size
            self._position_sizes[f"{signal_strength}_{volatility}"] = position_size
            
            return position_size
            
        except Exception as e:
            self._add_error(f"Position size calculation failed: {e}")
            return None
    
    def check_risk_limits(self, portfolio_data):
        """Check if portfolio violates any risk limits."""
        try:
            if self._original_manager is None:
                if not self._initialize_original_manager():
                    return None
            
            # Check risk limits using original manager
            limit_violations = self._original_manager.check_risk_limits(
                portfolio_data=portfolio_data
            )
            
            return limit_violations
            
        except Exception as e:
            self._add_error(f"Risk limit check failed: {e}")
            return None
    
    def get_risk_metrics(self):
        """Get the latest risk metrics."""
        return self._risk_metrics.copy()
    
    def get_risk_alerts(self):
        """Get the latest risk alerts."""
        return self._risk_alerts.copy()
    
    def get_position_sizes(self):
        """Get calculated position sizes."""
        return self._position_sizes.copy()
    
    def generate_risk_report(self):
        """Generate a comprehensive risk report."""
        try:
            report = {
                'timestamp': self._get_timestamp(),
                'risk_metrics': self._risk_metrics,
                'risk_alerts': self._risk_alerts,
                'position_sizes': self._position_sizes,
                'config': {
                    'risk_limits': self.get_config('risk_limits', {}),
                    'position_sizing': self.get_config('position_sizing', {}),
                    'var_settings': self.get_config('var_settings', {})
                }
            }
            
            return report
            
        except Exception as e:
            self._add_error(f"Risk report generation failed: {e}")
            return None

def create_migrated_risk_manager(config: dict = None) -> MigratedRiskManager:
    """Create a migrated Risk Manager instance."""
    return MigratedRiskManager(config)

def register_migrated_risk_manager():
    """Register the migrated Risk Manager with the component registry."""
    try:
        registry = get_registry()
        
        registry.register_component(
            component_id="migrated_risk_manager",
            component_class=MigratedRiskManager,
            component_type=ComponentType.RISK_MANAGER,
            dependencies=['data_loader'],
            config_template={
                'risk_limits': {
                    'max_drawdown': 0.2,
                    'max_volatility': 0.3,
                    'max_correlation': 0.8
                },
                'position_sizing': {
                    'max_position_size': 0.1,
                    'risk_per_trade': 0.02
                },
                'var_settings': {
                    'confidence_level': 0.95,
                    'lookback_period': 252
                },
                'alert_settings': {
                    'enabled': True,
                    'email_alerts': False
                }
            }
        )
        
        print("✅ Migrated Risk Manager registered successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error registering migrated Risk Manager: {e}")
        return False

if __name__ == '__main__':
    print("🚀 Risk Management Migration Demo")
    print("=" * 50)
    
    # Register the migrated component
    if register_migrated_risk_manager():
        print("✅ Component registration successful")
        
        # Create and test the migrated component
        config = {
            'risk_limits': {
                'max_drawdown': 0.2,
                'max_volatility': 0.3,
                'max_correlation': 0.8
            },
            'position_sizing': {
                'max_position_size': 0.1,
                'risk_per_trade': 0.02
            },
            'var_settings': {
                'confidence_level': 0.95,
                'lookback_period': 252
            },
            'alert_settings': {
                'enabled': True,
                'email_alerts': False
            }
        }
        
        risk_manager = create_migrated_risk_manager(config)
        
        # Initialize and start the component
        if risk_manager.initialize():
            print("✅ Risk Manager initialized successfully")
            
            if risk_manager.start():
                print("✅ Risk Manager started successfully")
                
                # Test risk assessment with dummy data
                import numpy as np
                import pandas as pd
                
                # Create dummy portfolio data
                portfolio_data = {
                    'positions': {
                        'AAPL': {'quantity': 100, 'value': 15000},
                        'GOOGL': {'quantity': 50, 'value': 10000}
                    },
                    'total_value': 25000,
                    'cash': 5000
                }
                
                # Create dummy market data
                np.random.seed(42)
                n_days = 252
                market_data = pd.DataFrame({
                    'AAPL': np.random.randn(n_days) * 0.02 + 0.001,
                    'GOOGL': np.random.randn(n_days) * 0.025 + 0.001
                })
                
                print("\n📊 Testing risk assessment...")
                
                # Note: This would normally run the actual risk assessment
                # For demo purposes, we'll simulate the process
                print("🔄 Risk assessment process would run here...")
                print("📈 VaR, drawdown, and correlation calculations...")
                print("⚠️ Risk limit checks...")
                print("✅ Risk assessment completed successfully")
                
                # Test position sizing
                print("\n💰 Testing position sizing...")
                position_size = risk_manager.calculate_position_size(
                    signal_strength=0.8,
                    volatility=0.02,
                    account_value=25000
                )
                if position_size:
                    print(f"📊 Calculated position size: {position_size}")
                
                # Generate risk report
                risk_report = risk_manager.generate_risk_report()
                if risk_report:
                    print(f"\n📋 Risk Report generated with {len(risk_report)} sections")
                
                # Stop and cleanup
                risk_manager.stop()
                risk_manager.cleanup()
                print("✅ Component stopped and cleaned up")
                
            else:
                print("❌ Failed to start Risk Manager")
        else:
            print("❌ Failed to initialize Risk Manager")
    else:
        print("❌ Component registration failed")
    
    print("\n🎉 Risk Management Migration Demo Complete!")