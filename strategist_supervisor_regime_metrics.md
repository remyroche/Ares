# Per-Regime Monitoring & Reporting for Strategist/Supervisor

## Strategist Per-Regime Metrics

```python
class RegimeAwareStrategistMetrics:
    def __init__(self):
        self.metrics_tracker = {
            'bull': RegimeMetricsTracker('bull'),
            'bear': RegimeMetricsTracker('bear'),
            'sideways': RegimeMetricsTracker('sideways'),
            'transition': RegimeMetricsTracker('transition')
        }
        self.performance_history = []
        
    async def track_decision(self, decision, regime, regime_confidence):
        """Track each strategist decision with regime context."""
        
        # Determine tracking category
        if regime_confidence < 0.75:
            tracking_regime = 'transition'
        else:
            tracking_regime = regime
            
        # Track decision
        await self.metrics_tracker[tracking_regime].track({
            'timestamp': decision['timestamp'],
            'action': decision['action'],
            'confidence': decision['confidence'],
            'regime_confidence': regime_confidence,
            'features_used': decision.get('features_used', []),
            'models_used': decision.get('models_used', [])
        })
        
    async def track_outcome(self, decision_id, outcome, regime):
        """Track outcome of decisions for P&L calculation."""
        tracker = self._get_appropriate_tracker(decision_id)
        
        await tracker.track_outcome({
            'decision_id': decision_id,
            'pnl': outcome['pnl'],
            'win': outcome['pnl'] > 0,
            'holding_period': outcome['holding_period'],
            'max_drawdown': outcome.get('max_drawdown', 0),
            'regime_at_exit': outcome.get('exit_regime', regime)
        })
        
    def generate_regime_report(self, period='daily'):
        """Generate comprehensive per-regime performance report."""
        report = {
            'period': period,
            'timestamp': datetime.now(),
            'regime_performance': {}
        }
        
        for regime, tracker in self.metrics_tracker.items():
            metrics = tracker.calculate_metrics(period)
            
            report['regime_performance'][regime] = {
                'decisions': {
                    'total': metrics['total_decisions'],
                    'long': metrics['long_decisions'],
                    'short': metrics['short_decisions'],
                    'hold': metrics['hold_decisions']
                },
                'performance': {
                    'total_pnl': metrics['total_pnl'],
                    'win_rate': metrics['win_rate'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown': metrics['max_drawdown'],
                    'profit_factor': metrics['profit_factor'],
                    'avg_holding_period': metrics['avg_holding_period']
                },
                'model_usage': metrics['model_usage'],
                'confidence_distribution': metrics['confidence_distribution']
            }
            
        # Add comparative analysis
        report['comparative_analysis'] = self._generate_comparative_analysis(report)
        
        return report
```

## Supervisor Per-Regime Monitoring

```python
class RegimeAwareSupervisor:
    def __init__(self, config):
        self.config = config
        self.regime_monitors = {
            'bull': RegimeMonitor('bull', config),
            'bear': RegimeMonitor('bear', config),
            'sideways': RegimeMonitor('sideways', config)
        }
        self.alert_thresholds = {
            'bull': {'min_sharpe': 0.8, 'max_drawdown': 0.15},
            'bear': {'min_sharpe': 0.5, 'max_drawdown': 0.20},
            'sideways': {'min_sharpe': 0.6, 'max_drawdown': 0.10}
        }
        
    async def supervise(self, strategist_decisions, market_data, regime_info):
        """Supervise strategist decisions with regime awareness."""
        
        current_regime = regime_info['current_regime']
        regime_confidence = regime_info['confidence']
        
        # Get appropriate monitor
        monitor = self.regime_monitors[current_regime]
        
        # Validate decision for current regime
        validation = await monitor.validate_decision(
            strategist_decisions,
            market_data,
            regime_confidence
        )
        
        # Check regime-specific risk limits
        risk_check = await self._check_regime_risk_limits(
            current_regime,
            strategist_decisions
        )
        
        # Generate supervision decision
        supervision_result = {
            'approved': validation['passed'] and risk_check['passed'],
            'regime': current_regime,
            'regime_confidence': regime_confidence,
            'validation_details': validation,
            'risk_check_details': risk_check,
            'adjustments': []
        }
        
        # Apply regime-specific adjustments if needed
        if not supervision_result['approved']:
            supervision_result['adjustments'] = self._generate_adjustments(
                current_regime,
                validation,
                risk_check
            )
            
        # Track supervision metrics
        await self._track_supervision_metrics(supervision_result)
        
        return supervision_result
    
    async def generate_regime_risk_report(self):
        """Generate risk report segmented by regime."""
        report = {
            'timestamp': datetime.now(),
            'regime_risk_metrics': {}
        }
        
        for regime, monitor in self.regime_monitors.items():
            risk_metrics = await monitor.calculate_risk_metrics()
            
            report['regime_risk_metrics'][regime] = {
                'current_exposure': risk_metrics['exposure'],
                'var_95': risk_metrics['var_95'],
                'cvar_95': risk_metrics['cvar_95'],
                'stress_test_results': risk_metrics['stress_tests'],
                'correlation_matrix': risk_metrics['correlations'],
                'regime_specific_risks': self._identify_regime_risks(regime, risk_metrics)
            }
            
        # Add cross-regime analysis
        report['cross_regime_analysis'] = {
            'regime_transition_risk': self._calculate_transition_risk(),
            'regime_correlation': self._calculate_regime_correlation(),
            'worst_case_scenario': self._calculate_worst_case()
        }
        
        return report
```

## Real-Time Regime Dashboard

```python
class RegimeDashboard:
    def __init__(self):
        self.real_time_metrics = {}
        self.update_frequency = 60  # Update every minute
        
    async def update_dashboard(self):
        """Update real-time regime metrics."""
        
        dashboard_data = {
            'current_regime': await self._get_current_regime(),
            'regime_confidence': await self._get_regime_confidence(),
            'regime_metrics': {}
        }
        
        # Get metrics for each regime
        for regime in ['bull', 'bear', 'sideways']:
            regime_data = await self._get_regime_metrics(regime)
            
            dashboard_data['regime_metrics'][regime] = {
                'active_positions': regime_data['positions'],
                'pnl_today': regime_data['pnl_today'],
                'pnl_mtd': regime_data['pnl_mtd'],
                'sharpe_30d': regime_data['sharpe_30d'],
                'win_rate_30d': regime_data['win_rate_30d'],
                'current_drawdown': regime_data['drawdown'],
                'regime_duration': regime_data['duration_hours'],
                'last_trade': regime_data['last_trade_time']
            }
            
        # Add alerts
        dashboard_data['alerts'] = await self._check_regime_alerts()
        
        # Add recommendations
        dashboard_data['recommendations'] = self._generate_recommendations(
            dashboard_data
        )
        
        return dashboard_data
```

## MLflow Integration for Regime Tracking

```python
class RegimeMLflowTracker:
    def __init__(self, config):
        self.mlflow_client = mlflow.tracking.MlflowClient()
        self.experiment_name = f"regime_tracking_{config['symbol']}"
        
    async def log_regime_metrics(self, regime, metrics):
        """Log regime-specific metrics to MLflow."""
        
        with mlflow.start_run(run_name=f"regime_{regime}_{datetime.now()}"):
            # Log metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{regime}_{metric_name}", value)
            
            # Log regime-specific parameters
            mlflow.log_param(f"{regime}_active", True)
            mlflow.log_param(f"{regime}_confidence", metrics.get('confidence', 0))
            
            # Log artifacts
            if 'report' in metrics:
                mlflow.log_dict(metrics['report'], f"{regime}_report.json")
```