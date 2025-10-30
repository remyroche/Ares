#!/usr/bin/env python3
"""
SR Workflow Runner

This script runs the complete SR (Support/Resistance) workflow in the correct order:
1. SR Parameter Optimization - Optimizes parameters for SR detection
2. SR Detection - Detects SR levels using optimized parameters
3. SR Clustering - Clusters the detected SR levels

Usage Examples:
    # Basic usage with defaults
    python scripts/run_sr_workflow.py --symbol ETHUSDT --exchange binance --timeframe 15m
    
    # With lookback period (days)
    python scripts/run_sr_workflow.py --symbol BTCUSDT --exchange binance --timeframe 1h --lookback-days 30
    
    # With explicit end date (start date calculated from lookback-days)
    python scripts/run_sr_workflow.py --symbol ETHUSDT --exchange binance --timeframe 15m --end-date 2024-01-31 --lookback-days 30
    
    # Full mode with all options
    python scripts/run_sr_workflow.py --symbol BTCUSDT --exchange binance --timeframe 1h --direction long --mode full --lookback-days 60
"""

import asyncio
import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger

# Import SR workflow steps
from src.training.steps.market_analysis.components.sr_parameter_optimization import SRParameterOptimizationStep
from src.training.steps.market_analysis.components.sr_detection import SRDetectionComponent
from src.training.steps.market_analysis.components.sr_clustering import SRClusteringComponent
from src.tactician.sr_levels.ml_quality import SRQualityDataCollector, SRQualityModel


class SRWorkflowRunner:
    """
    Runs the complete SR workflow:
    1. SR Parameter Optimization
    2. SR Detection
    3. SR Clustering
    """
    
    def __init__(
        self,
        symbol: str = "ETHUSDT",
        exchange: str = "binance",
        timeframe: str = "15m",
        direction: str = "long",
        mode: str = "light",
        end_date: Optional[str] = None,
        lookback_days: Optional[int] = None,
        # ML training controls
        train_ml: bool = False,
        ml_start_date: Optional[str] = None,
        ml_end_date: Optional[str] = None,
        ml_timeframe: Optional[str] = None,
        ml_model_output: str = "models/sr_quality_model.lgb",
        ml_sample_freq_days: int = 7,
        ml_forward_days: int = 10
    ):
        """
        Initialize the SR workflow runner.
        
        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            exchange: Exchange name (e.g., 'binance')
            timeframe: Timeframe (e.g., '15m', '1h', '1d')
            direction: Trading direction ('long' or 'short')
            mode: Execution mode ('light', 'full', or 'blank')
            end_date: End date for data range (YYYY-MM-DD format). If None, uses latest available data
            lookback_days: Number of days to look back from end_date. If provided, calculates start_date automatically
        """
        self.logger = system_logger.getChild('SRWorkflowRunner')
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.direction = direction
        self.mode = mode
        self.end_date = end_date
        self.lookback_days = lookback_days
        # ML training config
        self.train_ml = train_ml
        self.ml_start_date = ml_start_date
        self.ml_end_date = ml_end_date
        self.ml_timeframe = ml_timeframe or timeframe
        self.ml_model_output = ml_model_output
        self.ml_sample_freq_days = ml_sample_freq_days
        self.ml_forward_days = ml_forward_days
        
        # Calculate start_date from lookback_days if provided
        self.start_date = None
        if lookback_days:
            if end_date:
                try:
                    end = datetime.strptime(end_date, '%Y-%m-%d')
                    self.start_date = (end - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
                    self.logger.info(f"Calculated start_date from lookback_days={lookback_days} and end_date={end_date}: {self.start_date}")
                except ValueError:
                    self.logger.warning(f"Invalid end_date format: {end_date}, will use latest available data")
            else:
                # If end_date is None, we'll let the data loading components handle it
                # They will use latest available data and calculate start_date accordingly
                self.logger.info(f"Lookback_days={lookback_days} provided, will calculate start_date from latest available data")
                # Pass lookback_days to configs instead of calculating start_date upfront
        
        # Initialize steps
        self.logger.info("🚀 Initializing SR workflow steps...")
        self.param_optimizer = SRParameterOptimizationStep()
        self.sr_detector = SRDetectionComponent()
        self.sr_clusterer = SRClusteringComponent()
        
        # Workflow state
        self.workflow_state = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'direction': direction,
            'execution_mode': mode,
            'start_time': datetime.now()
        }
        
        # Create outcomes directory for reports
        self.outcomes_dir = Path('outcomes') / f"sr_workflow_{symbol}_{timeframe}"
        self.outcomes_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate datetime stamp for filenames
        self.datetime_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Store report paths
        self.report_paths = {}
        
        self.logger.info("✅ SR workflow runner initialized")
        self.logger.info(f"📁 Reports will be saved to: {self.outcomes_dir}")
    
    def _extract_optimized_parameters(self, optimization_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and validate optimized parameters from optimization result.
        
        Args:
            optimization_result: Result dictionary from parameter optimization step
            
        Returns:
            Dictionary containing optimized parameters, or empty dict if not found
        """
        try:
            # Parameters are stored in artifacts, not metrics
            artifacts = optimization_result.get('artifacts', {})
            if not artifacts:
                self.logger.warning("No artifacts found in optimization result")
                return {}
            
            # Extract from nested artifact structure
            optimization_result_data = artifacts.get('sr_parameter_optimization_result', {})
            if not optimization_result_data:
                self.logger.warning("No sr_parameter_optimization_result found in artifacts")
                return {}
            
            optimized_params = optimization_result_data.get('optimized_parameters', {})
            if not optimized_params or not isinstance(optimized_params, dict):
                self.logger.warning("No optimized_parameters found in optimization result data")
                return {}
            
            return optimized_params
            
        except Exception as e:
            self.logger.error(f"Failed to extract optimized parameters: {e}")
            return {}
    
    def _generate_step_report(self, step_name: str, step_result: Dict[str, Any], 
                             step_duration: float) -> str:
        """Generate detailed markdown report for a pipeline step.
        
        Returns:
            Path to the generated report file
        """
        try:
            report_filename = f"{step_name}_{self.symbol}_{self.timeframe}_{self.datetime_stamp}.md"
            report_path = self.outcomes_dir / report_filename
            
            # Build report content
            report_lines = [
                f"# {step_name.upper().replace('_', ' ')} Report",
                f"",
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Symbol:** {self.symbol}",
                f"**Exchange:** {self.exchange}",
                f"**Timeframe:** {self.timeframe}",
                f"**Direction:** {self.direction}",
                f"**Mode:** {self.mode}",
                f"",
                f"---",
                f"",
                f"## Execution Summary",
                f"",
                f"- **Status:** {'✅ Success' if step_result.get('success') else '❌ Failed'}",
                f"- **Duration:** {step_duration:.2f} seconds",
                f"- **Step:** {step_name}",
                f"",
            ]
            
            # Add metrics if available
            metrics = step_result.get('metrics', {})
            if metrics:
                report_lines.extend([
                    f"## Metrics",
                    f"",
                    "```json",
                    json.dumps(metrics, indent=2, default=str),
                    "```",
                    f"",
                ])
            
            # Add artifacts info
            artifacts = step_result.get('artifacts', {})
            if artifacts:
                report_lines.extend([
                    f"## Artifacts Created",
                    f"",
                ])
                for key, value in artifacts.items():
                    report_lines.append(f"- **{key}:** {value}")
                report_lines.append("")
            
            # Add step-specific details
            if step_name == 'ml_model_training':
                ml_artifacts = artifacts.get('ml_training', {})
                if ml_artifacts:
                    report_lines.extend([
                        f"## ML Model Training Details",
                        f"",
                        f"- **Training Data Path:** {ml_artifacts.get('training_data_path', 'N/A')}",
                        f"- **Model Path:** {ml_artifacts.get('model_path', 'N/A')}",
                        f"",
                        f"### Cross-Validation Metrics",
                        f"",
                        "```json",
                        json.dumps(ml_artifacts.get('metrics', {}), indent=2, default=str),
                        "```",
                        f"",
                    ])
            
            # Add errors if any
            if 'error' in step_result:
                report_lines.extend([
                    f"## Error",
                    f"",
                    f"```",
                    str(step_result['error']),
                    f"```",
                    f"",
                ])
            
            # Write report
            with open(report_path, 'w') as f:
                f.write('\n'.join(report_lines))
            
            self.logger.info(f"📄 Report saved: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate report for {step_name}: {e}")
            return ""
    
    def _generate_summary_report(self, workflow_results: Dict[str, Any]) -> str:
        """Generate comprehensive workflow summary report.
        
        Returns:
            Path to the summary report file
        """
        try:
            report_filename = f"workflow_summary_{self.symbol}_{self.timeframe}_{self.datetime_stamp}.md"
            report_path = self.outcomes_dir / report_filename
            
            # Calculate summary stats
            total_steps = len(workflow_results['steps_completed']) + len(workflow_results['steps_failed'])
            success_rate = (len(workflow_results['steps_completed']) / total_steps * 100) if total_steps > 0 else 0
            
            # Build comprehensive report
            report_lines = [
                f"# SR Workflow Summary Report",
                f"",
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Symbol:** {self.symbol}",
                f"**Exchange:** {self.exchange}",
                f"**Timeframe:** {self.timeframe}",
                f"**Direction:** {self.direction}",
                f"**Mode:** {self.mode}",
                f"",
                f"---",
                f"",
                f"## Workflow Execution Summary",
                f"",
                f"- **Total Duration:** {workflow_results.get('total_duration', 0):.2f} seconds",
                f"- **Steps Completed:** {len(workflow_results['steps_completed'])}/{total_steps}",
                f"- **Steps Failed:** {len(workflow_results['steps_failed'])}/{total_steps}",
                f"- **Success Rate:** {success_rate:.1f}%",
                f"- **Start Time:** {workflow_results.get('start_time', 'N/A')}",
                f"- **End Time:** {workflow_results.get('end_time', 'N/A')}",
                f"",
                f"## Steps Completed",
                f"",
            ]
            
            for step in workflow_results['steps_completed']:
                report_lines.append(f"✅ {step}")
            report_lines.append("")
            
            if workflow_results['steps_failed']:
                report_lines.extend([
                    f"## Steps Failed",
                    f"",
                ])
                for step in workflow_results['steps_failed']:
                    report_lines.append(f"❌ {step}")
                    if step in workflow_results['errors']:
                        report_lines.append(f"   Error: {workflow_results['errors'][step]}")
                report_lines.append("")
            
            # Add artifacts summary
            report_lines.extend([
                f"## Artifacts Created",
                f"",
            ])
            for step_name, artifacts in workflow_results['artifacts'].items():
                if artifacts:
                    report_lines.append(f"### {step_name}")
                    report_lines.append("")
                    for key, value in artifacts.items():
                        report_lines.append(f"- **{key}:** `{value}`")
                    report_lines.append("")
            
            # Add metrics summary
            if workflow_results['metrics']:
                report_lines.extend([
                    f"## Metrics Summary",
                    f"",
                    "```json",
                    json.dumps(workflow_results['metrics'], indent=2, default=str),
                    "```",
                    f"",
                ])
            
            # Add report paths
            report_lines.extend([
                f"## Individual Step Reports",
                f"",
            ])
            for step_name, report_path in self.report_paths.items():
                if step_name != 'workflow_summary':
                    report_lines.append(f"- [{step_name}]({report_path})")
            report_lines.append("")
            
            # Write summary report
            with open(report_path, 'w') as f:
                f.write('\n'.join(report_lines))
            
            self.logger.info(f"📄 Summary report saved: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")
            return ""
    
    async def run_workflow(self) -> Dict[str, Any]:
        """
        Run the complete SR workflow.
        
        Returns:
            Dict containing workflow results and artifacts
        """
        self.logger.info("=" * 80)
        self.logger.info("🎯 STARTING SR WORKFLOW")
        self.logger.info(f"   Symbol: {self.symbol}")
        self.logger.info(f"   Exchange: {self.exchange}")
        self.logger.info(f"   Timeframe: {self.timeframe}")
        self.logger.info(f"   Direction: {self.direction}")
        self.logger.info(f"   Mode: {self.mode}")
        if self.end_date:
            self.logger.info(f"   End Date: {self.end_date}")
        else:
            self.logger.info(f"   End Date: Latest available data")
        if self.lookback_days:
            if self.start_date:
                self.logger.info(f"   Lookback: {self.lookback_days} days (start_date: {self.start_date})")
            else:
                self.logger.info(f"   Lookback: {self.lookback_days} days (will use latest available data)")
        self.logger.info("=" * 80)
        
        workflow_results = {
            'success': False,
            'steps_completed': [],
            'steps_failed': [],
            'artifacts': {},
            'metrics': {},
            'errors': {}
        }
        
        try:
            # Optional Step 0: Train ML model (runs once before the rest)
            if self.train_ml:
                self.logger.info("\n" + "=" * 80)
                self.logger.info("🧠 STEP 0: TRAIN SR QUALITY ML MODEL")
                self.logger.info("=" * 80)
                if not self.ml_start_date or not self.ml_end_date:
                    raise ValueError("ML training requires --ml-start-date and --ml-end-date")
                self.logger.info(f"   ML Train Period: {self.ml_start_date} → {self.ml_end_date} | TF={self.ml_timeframe}")

                collector = SRQualityDataCollector()
                training_df = collector.collect_training_data(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    start_date=self.ml_start_date,
                    end_date=self.ml_end_date,
                    timeframe=self.ml_timeframe,
                    forward_days=self.ml_forward_days,
                    sample_freq_days=self.ml_sample_freq_days
                )
                saved_training_path = collector.save_training_data(training_df)
                self.logger.info(f"📦 Saved ML training dataset to: {saved_training_path}")

                model = SRQualityModel()
                metrics = model.train(training_df, target_column='quality_score', n_folds=5)
                model.save(self.ml_model_output)
                self.logger.info(f"✅ ML model saved to: {self.ml_model_output}")
                self.logger.info(f"📊 ML CV avg Val R²: {metrics['avg_metrics']['avg_val_r2']:.4f}")
                
                ml_duration = (datetime.now() - self.workflow_state['start_time']).total_seconds()
                workflow_results['steps_completed'].append('ml_model_training')
                workflow_results['artifacts']['ml_training'] = {
                    'training_data_path': saved_training_path,
                    'model_path': self.ml_model_output,
                    'metrics': metrics
                }
                
                # Generate ML training report
                ml_report_result = {
                    'success': True,
                    'metrics': metrics,
                    'artifacts': {'ml_training': workflow_results['artifacts']['ml_training']}
                }
                ml_report_path = self._generate_step_report('ml_model_training', ml_report_result, ml_duration)
                self.report_paths['ml_model_training'] = ml_report_path

            # Step 1: SR Parameter Optimization
            self.logger.info("\n" + "=" * 80)
            self.logger.info("📊 STEP 1: SR PARAMETER OPTIMIZATION")
            self.logger.info("=" * 80)
            
            optimization_config = {
                'symbol': self.symbol,
                'exchange': self.exchange,
                'timeframe': self.timeframe,
                'direction': self.direction,
                'execution_mode': self.mode,
                'enable_bayesian_hpo': True,
                'enable_vectorbt': True,
                'enable_hardware_optimization': True,
                'enable_sr_detection_testing': True,  # Enable SR detection testing during optimization
            }
            
            # Add date range parameters if provided
            if self.start_date:
                optimization_config['start_date'] = self.start_date
            if self.end_date:
                optimization_config['end_date'] = self.end_date
            if self.lookback_days and not self.start_date:
                # If lookback_days provided but no explicit start_date, pass lookback_days for component to handle
                optimization_config['lookback_days'] = self.lookback_days
            
            optimization_result = await self.param_optimizer.execute(optimization_config)
            
            if not optimization_result.get('success', False):
                error_msg = optimization_result.get('error', 'Unknown error in parameter optimization')
                self.logger.error(f"❌ Parameter optimization failed: {error_msg}")
                workflow_results['steps_failed'].append('sr_parameter_optimization')
                workflow_results['errors']['sr_parameter_optimization'] = error_msg
                return workflow_results
            
            self.logger.info("✅ Parameter optimization completed successfully")
            workflow_results['steps_completed'].append('sr_parameter_optimization')
            workflow_results['artifacts']['optimization'] = optimization_result.get('artifacts', {})
            workflow_results['metrics']['optimization'] = optimization_result.get('metrics', {})
            
            # Generate parameter optimization report
            opt_duration = (datetime.now() - self.workflow_state['start_time']).total_seconds()
            opt_report_path = self._generate_step_report('sr_parameter_optimization', optimization_result, opt_duration)
            self.report_paths['sr_parameter_optimization'] = opt_report_path
            
            # Extract optimized parameters from artifacts (not metrics)
            optimized_params = self._extract_optimized_parameters(optimization_result)
            if optimized_params:
                self.logger.info(f"📊 Loaded {len(optimized_params)} optimized parameters")
            else:
                self.logger.warning("⚠️ No optimized parameters found, detection will use defaults")
            
            # Step 2: SR Detection with optimized parameters
            self.logger.info("\n" + "=" * 80)
            self.logger.info("📊 STEP 2: SR DETECTION WITH OPTIMIZED PARAMETERS")
            self.logger.info("=" * 80)
            
            detection_config = {
                'symbol': self.symbol,
                'exchange': self.exchange,
                'timeframe': self.timeframe,
                'direction': self.direction,
                'execution_mode': self.mode,
                'enable_shap_lime': True,
                'enable_vectorbt': True,
                'enable_hardware_optimization': True,
                # Pass optimized parameters from step 1
                'sr_parameters': optimized_params
            }
            
            # Add date range parameters if provided
            if self.start_date:
                detection_config['start_date'] = self.start_date
            if self.end_date:
                detection_config['end_date'] = self.end_date
            if self.lookback_days and not self.start_date:
                # If lookback_days provided but no explicit start_date, pass lookback_days for component to handle
                detection_config['lookback_days'] = self.lookback_days
            
            detection_result = await self.sr_detector.execute(detection_config)
            
            if not detection_result.get('success', False):
                error_msg = detection_result.get('error', 'Unknown error in SR detection')
                self.logger.error(f"❌ SR detection failed: {error_msg}")
                workflow_results['steps_failed'].append('sr_detection')
                workflow_results['errors']['sr_detection'] = error_msg
                return workflow_results
            
            self.logger.info("✅ SR detection completed successfully")
            workflow_results['steps_completed'].append('sr_detection')
            workflow_results['artifacts']['detection'] = detection_result.get('artifacts', {})
            workflow_results['metrics']['detection'] = detection_result.get('metrics', {})
            
            # Generate SR detection report
            det_start = datetime.now()
            det_duration = (det_start - self.workflow_state['start_time']).total_seconds()
            det_report_path = self._generate_step_report('sr_detection', detection_result, det_duration)
            self.report_paths['sr_detection'] = det_report_path
            
            # Extract detected SR levels with validation
            sr_levels = detection_result.get('detection_result', {})
            if not sr_levels or not isinstance(sr_levels, dict):
                self.logger.warning("⚠️ No SR levels detected or invalid detection result format")
                sr_levels = {'total_levels': 0}
            total_levels = sr_levels.get('total_levels', 0)
            self.logger.info(f"📊 Detected {total_levels} SR levels")
            
            # Step 3: SR Clustering
            self.logger.info("\n" + "=" * 80)
            self.logger.info("📊 STEP 3: SR CLUSTERING")
            self.logger.info("=" * 80)
            
            clustering_config = {
                'symbol': self.symbol,
                'exchange': self.exchange,
                'timeframe': self.timeframe,
                'direction': self.direction,
                'execution_mode': self.mode,
                'enable_hardware_optimization': True,
                'enable_vectorbt_optimization': True,
                'enable_memory_optimization': True,
                'enable_gpu_acceleration': True,
                'clustering_algorithm': 'ensemble',  # Use ensemble for best results
                # SR levels will be loaded from artifacts automatically
            }
            
            # Add date range parameters if provided
            if self.start_date:
                clustering_config['start_date'] = self.start_date
            if self.end_date:
                clustering_config['end_date'] = self.end_date
            if self.lookback_days and not self.start_date:
                # If lookback_days provided but no explicit start_date, pass lookback_days for component to handle
                clustering_config['lookback_days'] = self.lookback_days
            
            clustering_result = await self.sr_clusterer.execute(clustering_config)
            
            if not clustering_result.get('success', False):
                error_msg = clustering_result.get('error', 'Unknown error in SR clustering')
                self.logger.error(f"❌ SR clustering failed: {error_msg}")
                workflow_results['steps_failed'].append('sr_clustering')
                workflow_results['errors']['sr_clustering'] = error_msg
                return workflow_results
            
            self.logger.info("✅ SR clustering completed successfully")
            workflow_results['steps_completed'].append('sr_clustering')
            workflow_results['artifacts']['clustering'] = clustering_result.get('artifacts', {})
            workflow_results['metrics']['clustering'] = clustering_result.get('metrics', {})
            
            # Generate SR clustering report
            clust_start = datetime.now()
            clust_duration = (clust_start - self.workflow_state['start_time']).total_seconds()
            clust_report_path = self._generate_step_report('sr_clustering', clustering_result, clust_duration)
            self.report_paths['sr_clustering'] = clust_report_path
            
            # Extract clustering results with validation
            clusters = clustering_result.get('clustering_result', {})
            if not clusters or not isinstance(clusters, dict):
                self.logger.warning("⚠️ No clusters created or invalid clustering result format")
                clusters = {'total_clusters': 0}
            total_clusters = clusters.get('total_clusters', 0)
            self.logger.info(f"📊 Created {total_clusters} SR clusters")
            
            # Mark workflow as successful
            workflow_results['success'] = True
            workflow_results['end_time'] = datetime.now()
            workflow_results['total_duration'] = (
                workflow_results['end_time'] - self.workflow_state['start_time']
            ).total_seconds()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("✅ SR WORKFLOW COMPLETED SUCCESSFULLY")
            # Calculate totals for summary
            total_steps = 3  # Total number of workflow steps
            param_count = len(optimized_params) if isinstance(optimized_params, dict) else 0
            self.logger.info(f"   Steps completed: {len(workflow_results['steps_completed'])}/{total_steps}")
            self.logger.info(f"   Total duration: {workflow_results['total_duration']:.2f}s")
            self.logger.info(f"   Optimized parameters: {param_count} params")
            self.logger.info(f"   Detected SR levels: {total_levels}")
            self.logger.info(f"   Created clusters: {total_clusters}")
            
            # Generate final summary report
            summary_report_path = self._generate_summary_report(workflow_results)
            self.report_paths['workflow_summary'] = summary_report_path
            
            self.logger.info("\n📄 Reports Generated:")
            for step_name, report_path in self.report_paths.items():
                self.logger.info(f"   {step_name}: {report_path}")
            self.logger.info("=" * 80)
            
            return workflow_results
            
        except Exception as e:
            self.logger.error(f"❌ Workflow failed with exception: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            workflow_results['errors']['workflow'] = str(e)
            workflow_results['end_time'] = datetime.now()
            return workflow_results


async def main():
    """Main entry point for the SR workflow runner."""
    parser = argparse.ArgumentParser(
        description="Run the complete SR workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (uses latest available data)
  python scripts/run_sr_workflow.py --symbol ETHUSDT --exchange binance --timeframe 15m
  
  # With 30-day lookback from latest available data
  python scripts/run_sr_workflow.py --symbol BTCUSDT --exchange binance --timeframe 1h --lookback-days 30
  
  # With explicit end date and lookback
  python scripts/run_sr_workflow.py --symbol ETHUSDT --exchange binance --timeframe 15m --end-date 2024-01-31 --lookback-days 30
        """
    )
    parser.add_argument('--symbol', type=str, default='ETHUSDT', help='Trading symbol (e.g., ETHUSDT, BTCUSDT)')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange name (e.g., binance, bybit)')
    parser.add_argument('--timeframe', type=str, default='15m', help='Timeframe (e.g., 15m, 1h, 4h, 1d)')
    parser.add_argument('--direction', type=str, default='long', choices=['long', 'short'], help='Trading direction')
    parser.add_argument('--mode', type=str, default='light', choices=['light', 'full', 'blank'], help='Execution mode')
    
    # Date range options
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for data range (YYYY-MM-DD format). If not provided, uses latest available data')
    parser.add_argument('--lookback-days', type=int, default=None,
                       help='Number of days to look back from end_date. Calculates start_date automatically')
    
    # Optional ML training stage
    parser.add_argument('--train-ml', action='store_true', help='Run ML training stage before the SR workflow')
    parser.add_argument('--ml-start-date', type=str, default=None, help='ML training start date (YYYY-MM-DD)')
    parser.add_argument('--ml-end-date', type=str, default=None, help='ML training end date (YYYY-MM-DD)')
    parser.add_argument('--ml-timeframe', type=str, default=None, help='ML training timeframe (defaults to --timeframe)')
    parser.add_argument('--ml-model-output', type=str, default='models/sr_quality_model.lgb', help='Path to save trained ML model')
    parser.add_argument('--ml-sample-freq-days', type=int, default=7, help='Sampling frequency in days for ML training data collection')
    parser.add_argument('--ml-forward-days', type=int, default=10, help='Forward window (days) to measure SR performance')
    
    args = parser.parse_args()
    
    # Validate date arguments
    if args.end_date:
        try:
            datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError:
            print(f"Error: Invalid end-date format: {args.end_date}. Expected YYYY-MM-DD")
            return 1
    
    if args.lookback_days and args.lookback_days <= 0:
        print(f"Error: lookback-days must be positive, got: {args.lookback_days}")
        return 1
    
    # Create workflow runner
    runner = SRWorkflowRunner(
        symbol=args.symbol,
        exchange=args.exchange,
        timeframe=args.timeframe,
        direction=args.direction,
        mode=args.mode,
        end_date=args.end_date,
        lookback_days=args.lookback_days,
        train_ml=args.train_ml,
        ml_start_date=args.ml_start_date,
        ml_end_date=args.ml_end_date,
        ml_timeframe=args.ml_timeframe,
        ml_model_output=args.ml_model_output,
        ml_sample_freq_days=args.ml_sample_freq_days,
        ml_forward_days=args.ml_forward_days
    )
    
    # Run workflow
    results = await runner.run_workflow()
    
    # Print summary
    if results['success']:
        print("\n" + "=" * 80)
        print("✅ SR WORKFLOW COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Steps completed: {', '.join(results['steps_completed'])}")
        print(f"Total duration: {results.get('total_duration', 0):.2f}s")
        print("\nArtifacts created:")
        for step_name, artifacts in results.get('artifacts', {}).items():
            print(f"  {step_name}: {len(artifacts)} artifacts")
        return 0
    else:
        print("\n" + "=" * 80)
        print("❌ SR WORKFLOW FAILED")
        print("=" * 80)
        print(f"Steps completed: {', '.join(results['steps_completed'])}")
        print(f"Steps failed: {', '.join(results['steps_failed'])}")
        print("\nErrors:")
        for step_name, error in results.get('errors', {}).items():
            print(f"  {step_name}: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
