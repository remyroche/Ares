"""
Unified Step08 Final Methods - Part 5
"""

    async def _generate_comprehensive_results(self, data: pd.DataFrame, selected_features: Dict[str, List[str]], 
                                            financial_metrics: FinancialMetrics, risk_metrics: RiskMetrics,
                                            feature_validation: FeatureSelectionValidation, start_time: datetime) -> Step08Results:
        """Generate comprehensive results from all analysis components."""
        try:
            self.logger.info('📋 Generating comprehensive results...')
            
            results = Step08Results()
            
            # Set basic results
            results.regime_data = data
            results.selected_features = selected_features
            results.financial_metrics = financial_metrics
            results.risk_metrics = risk_metrics
            results.regime_balance = self.regime_balance
            results.feature_validation = feature_validation
            
            # Execution metadata
            end_time = datetime.now()
            results.execution_metadata = {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': (end_time - start_time).total_seconds(),
                'total_samples': len(data),
                'total_features': len(data.columns),
                'selected_features_count': len(selected_features.get('final', [])),
                'regime_count': len(data['composite_cluster_id'].unique()) if 'composite_cluster_id' in data.columns else 0,
                'optimization_used': ENHANCED_OPTIMIZATIONS_AVAILABLE,
                'dependencies_available': {
                    'boruta': BORUTA_AVAILABLE,
                    'shap': SHAP_AVAILABLE,
                    'lime': LIME_AVAILABLE,
                    'numba': NUMBA_AVAILABLE,
                    'joblib': JOBLIB_AVAILABLE
                }
            }
            
            # Success determination
            results.success = (
                len(selected_features.get('final', [])) > 0 and
                feature_validation.validation_passed and
                risk_metrics.overall_risk_score < 0.8 and
                self.regime_balance.balance_score > 0.3
            )
            
            # Generate warnings
            if not feature_validation.validation_passed:
                results.warnings.append("Feature selection validation failed")
            if risk_metrics.overall_risk_score > 0.8:
                results.warnings.append("High overall risk score detected")
            if self.regime_balance.balance_score < 0.3:
                results.warnings.append("Poor regime balance detected")
            if not results.success:
                results.errors.append("Overall execution failed validation criteria")
            
            self.logger.info(f'✅ Comprehensive results generated:')
            self.logger.info(f'   Success: {results.success}')
            self.logger.info(f'   Selected features: {len(selected_features.get("final", []))}')
            self.logger.info(f'   Risk score: {risk_metrics.overall_risk_score:.3f}')
            self.logger.info(f'   Balance score: {self.regime_balance.balance_score:.3f}')
            
            return results
            
        except Exception as e:
            self.logger.error(f'Failed to generate comprehensive results: {e}')
            return Step08Results(success=False, errors=[str(e)])

    async def _save_artifacts_and_reports(self, results: Step08Results) -> None:
        """Save all artifacts and reports."""
        try:
            self.logger.info('💾 Saving artifacts and reports...')
            
            # Save regime data
            if results.regime_data is not None:
                regime_file = os.path.join(self.artifacts_dir, 'regime_data.parquet')
                results.regime_data.to_parquet(regime_file)
                results.artifacts_generated.append(regime_file)
            
            # Save selected features
            if results.selected_features:
                features_file = os.path.join(self.artifacts_dir, 'selected_features.json')
                safe_json_dump(results.selected_features, features_file)
                results.artifacts_generated.append(features_file)
            
            # Save financial metrics
            if results.financial_metrics:
                financial_file = os.path.join(self.metrics_dir, 'financial_metrics.json')
                safe_json_dump(results.financial_metrics.__dict__, financial_file)
                results.artifacts_generated.append(financial_file)
            
            # Save risk metrics
            if results.risk_metrics:
                risk_file = os.path.join(self.metrics_dir, 'risk_metrics.json')
                safe_json_dump(results.risk_metrics.__dict__, risk_file)
                results.artifacts_generated.append(risk_file)
            
            # Save regime balance metrics
            if results.regime_balance:
                balance_file = os.path.join(self.metrics_dir, 'regime_balance.json')
                safe_json_dump(results.regime_balance.__dict__, balance_file)
                results.artifacts_generated.append(balance_file)
            
            # Save feature validation
            if results.feature_validation:
                validation_file = os.path.join(self.metrics_dir, 'feature_validation.json')
                safe_json_dump(results.feature_validation.__dict__, validation_file)
                results.artifacts_generated.append(validation_file)
            
            # Save execution metadata
            if results.execution_metadata:
                metadata_file = os.path.join(self.reports_dir, 'execution_metadata.json')
                safe_json_dump(results.execution_metadata, metadata_file)
                results.artifacts_generated.append(metadata_file)
            
            # Generate comprehensive report
            comprehensive_report = self._generate_comprehensive_report(results)
            report_file = os.path.join(self.reports_dir, 'comprehensive_report.json')
            safe_json_dump(comprehensive_report, report_file)
            results.artifacts_generated.append(report_file)
            
            # Generate markdown report
            markdown_report = self._generate_markdown_report(results)
            markdown_file = os.path.join(self.reports_dir, 'comprehensive_report.md')
            with open(markdown_file, 'w') as f:
                f.write(markdown_report)
            results.artifacts_generated.append(markdown_file)
            
            # Generate visualizations
            await self._generate_visualizations(results)
            
            self.logger.info(f'✅ Artifacts and reports saved: {len(results.artifacts_generated)} files')
            
        except Exception as e:
            self.logger.error(f'Failed to save artifacts and reports: {e}')

    def _generate_comprehensive_report(self, results: Step08Results) -> Dict[str, Any]:
        """Generate comprehensive JSON report."""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'step_name': 'step08_unified',
                'version': '2.0.0',
                'summary': {
                    'success': results.success,
                    'total_samples': len(results.regime_data) if results.regime_data is not None else 0,
                    'selected_features_count': len(results.selected_features.get('final', [])),
                    'regime_count': len(results.regime_data['composite_cluster_id'].unique()) if results.regime_data is not None and 'composite_cluster_id' in results.regime_data.columns else 0,
                    'execution_time_seconds': results.execution_metadata.get('duration_seconds', 0),
                    'overall_risk_score': results.risk_metrics.overall_risk_score,
                    'balance_score': results.regime_balance.balance_score,
                    'validation_passed': results.feature_validation.validation_passed
                },
                'financial_metrics': results.financial_metrics.__dict__,
                'risk_metrics': results.risk_metrics.__dict__,
                'regime_balance': results.regime_balance.__dict__,
                'feature_validation': results.feature_validation.__dict__,
                'execution_metadata': results.execution_metadata,
                'selected_features': results.selected_features,
                'warnings': results.warnings,
                'errors': results.errors,
                'artifacts_generated': results.artifacts_generated
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f'Failed to generate comprehensive report: {e}')
            return {'error': str(e)}

    def _generate_markdown_report(self, results: Step08Results) -> str:
        """Generate comprehensive markdown report."""
        try:
            report = f"""# Step08 Unified Execution Report

## Executive Summary

- **Status**: {'✅ SUCCESS' if results.success else '❌ FAILED'}
- **Execution Time**: {results.execution_metadata.get('duration_seconds', 0):.2f} seconds
- **Total Samples**: {len(results.regime_data) if results.regime_data is not None else 0:,}
- **Selected Features**: {len(results.selected_features.get('final', []))}
- **Regime Count**: {len(results.regime_data['composite_cluster_id'].unique()) if results.regime_data is not None and 'composite_cluster_id' in results.regime_data.columns else 0}

## Financial Metrics

### Returns
- **Daily**: {results.financial_metrics.returns.get('daily', 0):.4f}
- **Annualized**: {results.financial_metrics.returns.get('annualized', 0):.4f}

### Risk Metrics
- **Volatility (Annualized)**: {results.financial_metrics.volatility.get('annualized', 0):.4f}
- **Sharpe Ratio**: {results.financial_metrics.sharpe_ratio.get('overall', 0):.4f}
- **Maximum Drawdown**: {results.financial_metrics.max_drawdown.get('overall', 0):.4f}
- **VaR (95%)**: {results.financial_metrics.var_95.get('overall', 0):.4f}

## Risk Assessment

- **Overall Risk Score**: {results.risk_metrics.overall_risk_score:.4f}
- **Portfolio VaR**: {results.risk_metrics.portfolio_var:.4f}
- **Model Risk**: {results.risk_metrics.model_risk:.4f}
- **Regime Risk**: {results.risk_metrics.regime_risk:.4f}
- **Overfitting Risk**: {results.risk_metrics.overfitting_risk:.4f}

## Regime Balance

- **Balance Score**: {results.regime_balance.balance_score:.4f}
- **Imbalance Severity**: {results.regime_balance.imbalance_severity}
- **Rebalancing Applied**: {'Yes' if results.regime_balance.rebalancing_applied else 'No'}

## Feature Selection Validation

- **Validation Passed**: {'Yes' if results.feature_validation.validation_passed else 'No'}
- **Selection Bias Score**: {results.feature_validation.selection_bias_score:.4f}
- **Temporal Stability**: {results.feature_validation.temporal_stability:.4f}
- **Regime Consistency**: {results.feature_validation.regime_consistency:.4f}

## Warnings and Errors

### Warnings
{chr(10).join(f'- {warning}' for warning in results.warnings) if results.warnings else 'None'}

### Errors
{chr(10).join(f'- {error}' for error in results.errors) if results.errors else 'None'}

## Generated Artifacts

{chr(10).join(f'- {artifact}' for artifact in results.artifacts_generated)}

---
*Report generated on: {datetime.now().isoformat()}*
"""
            
            return report
            
        except Exception as e:
            self.logger.error(f'Failed to generate markdown report: {e}')
            return f"# Error\n\nFailed to generate report: {e}"

    async def _generate_visualizations(self, results: Step08Results) -> None:
        """Generate visualization artifacts."""
        try:
            self.logger.info('📊 Generating visualizations...')
            
            # Set up matplotlib
            plt.style.use('default')
            fig_size = (12, 8)
            
            # 1. Regime distribution
            if results.regime_data is not None and 'composite_cluster_id' in results.regime_data.columns:
                fig, ax = plt.subplots(figsize=fig_size)
                regime_counts = results.regime_data['composite_cluster_id'].value_counts().sort_index()
                regime_counts.plot(kind='bar', ax=ax)
                ax.set_title('Regime Distribution')
                ax.set_xlabel('Regime ID')
                ax.set_ylabel('Count')
                plt.tight_layout()
                
                regime_plot_file = os.path.join(self.reports_dir, 'regime_distribution.png')
                plt.savefig(regime_plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                results.artifacts_generated.append(regime_plot_file)
            
            # 2. Financial metrics visualization
            if results.financial_metrics:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                
                # Returns
                returns_data = results.financial_metrics.returns
                if returns_data:
                    ax1.bar(returns_data.keys(), returns_data.values())
                    ax1.set_title('Returns by Period')
                    ax1.set_ylabel('Return')
                
                # Volatility
                volatility_data = results.financial_metrics.volatility
                if volatility_data:
                    ax2.bar(volatility_data.keys(), volatility_data.values())
                    ax2.set_title('Volatility by Period')
                    ax2.set_ylabel('Volatility')
                
                # Risk metrics
                risk_data = {
                    'Portfolio VaR': results.risk_metrics.portfolio_var,
                    'Model Risk': results.risk_metrics.model_risk,
                    'Regime Risk': results.risk_metrics.regime_risk,
                    'Overfitting Risk': results.risk_metrics.overfitting_risk
                }
                ax3.bar(risk_data.keys(), risk_data.values())
                ax3.set_title('Risk Metrics')
                ax3.set_ylabel('Risk Score')
                ax3.tick_params(axis='x', rotation=45)
                
                # Feature validation
                validation_data = {
                    'Selection Bias': results.feature_validation.selection_bias_score,
                    'Temporal Stability': results.feature_validation.temporal_stability,
                    'Regime Consistency': results.feature_validation.regime_consistency,
                    'Correlation Stability': results.feature_validation.correlation_stability
                }
                ax4.bar(validation_data.keys(), validation_data.values())
                ax4.set_title('Feature Validation Scores')
                ax4.set_ylabel('Score')
                ax4.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                metrics_plot_file = os.path.join(self.reports_dir, 'metrics_visualization.png')
                plt.savefig(metrics_plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                results.artifacts_generated.append(metrics_plot_file)
            
            # 3. Feature selection results
            if results.selected_features:
                fig, ax = plt.subplots(figsize=fig_size)
                feature_counts = {k: len(v) for k, v in results.selected_features.items()}
                ax.bar(feature_counts.keys(), feature_counts.values())
                ax.set_title('Feature Selection Results')
                ax.set_xlabel('Selection Phase')
                ax.set_ylabel('Number of Features')
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
                
                features_plot_file = os.path.join(self.reports_dir, 'feature_selection.png')
                plt.savefig(features_plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                results.artifacts_generated.append(features_plot_file)
            
            self.logger.info('✅ Visualizations generated successfully')
            
        except Exception as e:
            self.logger.error(f'Failed to generate visualizations: {e}')

# Decorated execution function
@deterministic_seed(42)
@idempotent_step(step_key='step08_unified')
@artifact_write_lock()
@nan_inf_and_constant_guard()
@artifact_versioning('2.0.0')
@time_budget_watchdog(soft_timeout_seconds=3600.0)
@validate_step_prerequisites(
    required_directories=['data/training', 'data_cache'],
    min_memory_gb=8.0,
    min_disk_gb=5.0,
    required_packages=['pandas', 'numpy', 'scipy'],
    data_quality_checks={
        'min_rows': 1000,
        'required_columns': ['timestamp', 'composite_cluster_id']
    },
    context='Unified Step08 Execution'
)
@secure_data_processing(
    backup_before=True,
    integrity_checks=True,
    memory_cleanup=True,
    data_validation=True
)
@prevent_data_leakage(
    temporal_validation=True,
    feature_leakage_detection=True,
    lookahead_bias_prevention=True
)
@resource_monitor(
    memory_threshold_gb=16.0,
    cpu_threshold_percent=80.0,
    disk_threshold_gb=10.0,
    monitor_interval=30.0,
    auto_cleanup=True
)
@memory_efficient(
    chunk_size=50000,
    streaming_processing=True,
    memory_pool=True,
    cleanup_frequency=50
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=True,
    performance_profiling=True,
    error_context_preservation=True
)
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=120.0,
    expected_exception=Exception,
    monitor_interval=30.0
)
@validate_step_output(
    required_files=['data/step08_unified/artifacts/regime_data.parquet'],
    data_quality_checks={
        'min_rows': 100,
        'required_columns': ['timestamp', 'composite_cluster_id']
    },
    performance_thresholds={'execution_time_minutes': 60.0},
    format_validation=True
)
@quality_gate(
    data_quality_metrics={'completeness': 0.95, 'consistency': 0.9},
    validation_score_requirements={'overall_risk_score': 0.8, 'balance_score': 0.3}
)
@auto_fix_data_quality_issues
@handle_errors(exceptions=(Exception,), default_return=False, context='step08_unified')
async def run_step(symbol: str, exchange: str, data_dir: str, timeframe: str = '1m', force_rerun: bool = False, **kwargs) -> bool:
    """Run unified Step08 with comprehensive analysis."""
    try:
        config = {
            'symbol': symbol,
            'exchange': exchange,
            'data_dir': data_dir,
            'timeframe': timeframe,
            'force_rerun': force_rerun,
            **kwargs
        }
        
        step = UnifiedStep08(config)
        result = await step.execute()
        
        return result.get('success', False)
        
    except Exception as e:
        system_logger.error(f'❌ Error running unified Step08: {e}')
        return False

if __name__ == '__main__':
    async def _test():
        await run_step('ETHUSDT', 'BINANCE', 'data/training')
    
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_test())
    except RuntimeError:
        asyncio.run(_test())