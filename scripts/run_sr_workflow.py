#!/usr/bin/env python3
"""
SR Workflow Runner

This script runs the complete SR (Support/Resistance) workflow in the correct order:
0. ML Model Training - Train ML model for SR quality scoring (default enabled)
0b. ML Model Validation - Validate model ranking performance with Precision@K, Spearman, etc.
1. SR Parameter Optimization - Optimizes parameters for SR detection using hierarchical HPO
2. SR Detection - Detects SR levels using optimized parameters and ML scoring
3. SR Filtering - Removes weak levels based on strength threshold (replaces clustering)

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
from typing import Dict, Any, Optional, List
from datetime import datetime as dt, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger

# Import enhanced configuration loader
from scripts.apply_enhanced_sr_optimization import (
    load_enhanced_config,
    create_enhanced_sr_config_dataclass
)

# Import SR workflow steps
from src.training.steps.market_analysis.components.sr_parameter_optimization import SRParameterOptimizationStep
from src.tactician.sr_levels.enhanced_sr_detection import EnhancedSRDetector
from src.training.steps.market_analysis.components.sr_clustering import SRClusteringComponent
from src.tactician.sr_levels.ml_quality import SRQualityDataCollector, SRQualityModel
from src.utils.data.real_data_loader import RealDataLoader

# Import validation functions
from scripts.validate_sr_ranking_metrics import (
    validate_ranking_metrics,
    print_ranking_results
)


class SRWorkflowRunner:
    """
    Runs the complete SR workflow:
    0. ML Model Training (optional, enabled by default)
    0b. ML Model Validation (validates ranking metrics)
    1. SR Parameter Optimization
    2. SR Detection (with ML scoring)
    3. SR Filtering (strength-based)
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
        # ML training controls (enabled by default)
        train_ml: bool = True,
        ml_start_date: Optional[str] = None,
        ml_end_date: Optional[str] = None,
        ml_timeframe: Optional[str] = None,
        ml_model_output: str = "models/sr_quality_model.lgb",
        ml_sample_freq_days: float = 0.5,  # 12-hour sampling for more training data (was 1 = daily)
        ml_forward_days: int = 10,
        enable_shap_reporting: bool = True  # Generate SHAP explanations in outcomes
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
        self.enable_shap_reporting = enable_shap_reporting
        
        # Calculate start_date from lookback_days if provided
        self.start_date = None
        if lookback_days:
            if end_date:
                try:
                    end = dt.strptime(end_date, '%Y-%m-%d')
                    self.start_date = (end - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
                    self.logger.info(f"Calculated start_date from lookback_days={lookback_days} and end_date={end_date}: {self.start_date}")
                except ValueError:
                    self.logger.warning(f"Invalid end_date format: {end_date}, will use latest available data")
            else:
                # If end_date is None, we'll let the data loading components handle it
                # They will use latest available data and calculate start_date accordingly
                self.logger.info(f"Lookback_days={lookback_days} provided, will calculate start_date from latest available data")
                # Pass lookback_days to configs instead of calculating start_date upfront
        
        # Load enhanced SR optimization configuration
        self.logger.info("📊 Loading enhanced SR optimization configuration...")
        try:
            enhanced_config_dict = load_enhanced_config()
            self.enhanced_sr_config = create_enhanced_sr_config_dataclass(enhanced_config_dict)
            self.logger.info(f"✅ Enhanced config loaded: {self.enhanced_sr_config.n_trials} trials, {self.enhanced_sr_config.optimization_level} mode")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load enhanced config, using defaults: {e}")
            self.enhanced_sr_config = None
        
        # Initialize steps
        self.logger.info("🚀 Initializing SR workflow steps...")
        self.param_optimizer = SRParameterOptimizationStep()
        # Note: SR clustering component removed from workflow - using strength-based filtering instead
        # Initialize EnhancedSRDetector with ML model support
        # Note: Build config with mode-aware optimizations
        sr_detector_config = {
            'min_touches': 2,
            'touch_proximity_threshold': 0.005,
            'min_strength': 0.15,
            'use_ml_model': True,  # Enable ML-based scoring
            'ml_model_path': ml_model_output if train_ml else 'models/sr_quality_model.lgb',
            
            # 🔧 CLUSTERING DISABLED: Essential for ML-based workflows
            'disable_dbscan_clustering': True,  # CRITICAL: Disable all clustering
            'disable_backtesting_validation': True,  # CRITICAL: Disable backtesting validation
            
            # 🚀 PERFORMANCE OPTIMIZATIONS
            'max_levels_per_method': 20,  # Reduce from 30 (33% faster)
            'max_fractal_levels': 20,      # Reduce from 30
            'max_pivot_levels': 20,        # Reduce from 30
            'max_volume_levels': 25,       # Reduce from 40
            
            # Use single period for faster detection (light mode optimization)
            'fractal_period': 5 if mode == 'light' else 3,  # Single period in light mode
            'pivot_period': 5 if mode == 'light' else 4,    # Single period in light mode
            
            # Enable caching for repeated detections
            'enable_fractal_caching': True,
            'enable_pivot_caching': True,
            
            # Reduce iterations for light mode (disable expensive methods)
            'psychological_levels': (mode == 'full'),  # Only in full mode
            'fibonacci_levels': (mode == 'full'),      # Only in full mode
        }
        self.sr_detector = EnhancedSRDetector(sr_detector_config)
        # self.sr_clusterer = SRClusteringComponent()  # REMOVED: Clustering not needed with ML model
        self.data_loader = RealDataLoader()
        
        # Workflow state
        self.workflow_state = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'direction': direction,
            'execution_mode': mode,
            'start_time': dt.now()
        }
        
        # Create outcomes directory for reports
        self.outcomes_dir = Path('outcomes') / f"sr_workflow_{symbol}_{timeframe}"
        self.outcomes_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate datetime stamp for filenames
        self.datetime_stamp = dt.now().strftime('%Y%m%d_%H%M%S')
        
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
            import pandas as pd
            
            # Parameters are stored in artifacts as file paths, not metrics
            artifacts = optimization_result.get('artifacts', {})
            if not artifacts:
                self.logger.warning("No artifacts found in optimization result")
                return {}
            
            # Get the file path to the sr_parameter_optimization_result artifact
            artifact_path = artifacts.get('sr_parameter_optimization_result', '')
            if not artifact_path or not isinstance(artifact_path, str):
                self.logger.warning(f"No sr_parameter_optimization_result path found in artifacts: {artifacts.keys()}")
                return {}
            
            # Load the parquet file
            from pathlib import Path
            path = Path(artifact_path)
            if not path.exists():
                self.logger.warning(f"Artifact file not found: {artifact_path}")
                return {}
            
            # Read parquet file
            df = pd.read_parquet(path)
            if df.empty:
                self.logger.warning(f"Artifact file is empty: {artifact_path}")
                return {}
            
            self.logger.info(f"📊 Loaded optimization artifact with {len(df)} rows and {len(df.columns)} columns")
            
            # Extract optimized_parameters from flattened column names
            # Columns are like: optimized_parameters.min_touches, optimized_parameters.strength_threshold, etc.
            optimized_params = {}
            for col in df.columns:
                if col.startswith('optimized_parameters.'):
                    param_name = col.replace('optimized_parameters.', '')
                    param_value = df[col].iloc[0]
                    optimized_params[param_name] = param_value
            
            if not optimized_params:
                self.logger.warning("No optimized_parameters columns found in parquet file")
                self.logger.info(f"Available columns: {list(df.columns)[:10]}")  # Show first 10
                return {}
            
            self.logger.info(f"✅ Extracted {len(optimized_params)} optimized parameters")
            self.logger.debug(f"Parameters: {optimized_params}")
            return optimized_params
            
        except Exception as e:
            self.logger.error(f"Failed to extract optimized parameters: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}
    
    def _analyze_sr_detection_methods(self, sr_levels: List[Any]) -> Dict[str, Any]:
        """Analyze which detection methods found strong SR levels."""
        try:
            method_stats = {}
            
            for level in sr_levels:
                # Extract method from metadata
                if hasattr(level, 'metadata'):
                    metadata = level.metadata or {}
                elif isinstance(level, dict):
                    metadata = level.get('metadata', {})
                else:
                    continue
                
                method = metadata.get('method', 'unknown')
                period = metadata.get('period', None)
                
                # Get level quality metrics
                strength = getattr(level, 'strength', None) if hasattr(level, 'strength') else level.get('strength', 0)
                level_type = getattr(level, 'type', None) if hasattr(level, 'type') else level.get('type', 'unknown')
                
                # Initialize method stats
                if method not in method_stats:
                    method_stats[method] = {
                        'count': 0,
                        'support_count': 0,
                        'resistance_count': 0,
                        'avg_strength': [],
                        'periods_used': set(),
                        'strongest_level': 0.0
                    }
                
                # Update stats
                method_stats[method]['count'] += 1
                method_stats[method]['avg_strength'].append(strength)
                method_stats[method]['strongest_level'] = max(method_stats[method]['strongest_level'], strength)
                
                if level_type == 'support':
                    method_stats[method]['support_count'] += 1
                elif level_type == 'resistance':
                    method_stats[method]['resistance_count'] += 1
                
                if period:
                    method_stats[method]['periods_used'].add(period)
            
            # Calculate averages and format
            method_analysis = {}
            for method, stats in method_stats.items():
                method_analysis[method] = {
                    'total_levels': stats['count'],
                    'support_levels': stats['support_count'],
                    'resistance_levels': stats['resistance_count'],
                    'avg_strength': sum(stats['avg_strength']) / len(stats['avg_strength']) if stats['avg_strength'] else 0.0,
                    'strongest_level': stats['strongest_level'],
                    'periods': sorted(list(stats['periods_used'])) if stats['periods_used'] else [],
                    'effectiveness_score': (sum(stats['avg_strength']) / len(stats['avg_strength']) if stats['avg_strength'] else 0.0) * 100
                }
            
            # Sort by effectiveness
            sorted_methods = sorted(method_analysis.items(), 
                                   key=lambda x: x[1]['effectiveness_score'], 
                                   reverse=True)
            
            return {
                'method_analysis': dict(sorted_methods),
                'total_levels': len(sr_levels),
                'total_methods': len(method_stats),
                'most_effective_method': sorted_methods[0][0] if sorted_methods else 'unknown'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze detection methods: {e}")
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
                f"**Generated:** {dt.now().strftime('%Y-%m-%d %H:%M:%S')}",
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
                # Handle both dict and list artifacts
                if isinstance(artifacts, dict):
                    for key, value in artifacts.items():
                        report_lines.append(f"- **{key}:** {value}")
                elif isinstance(artifacts, list):
                    for item in artifacts:
                        report_lines.append(f"- `{item}`")
                else:
                    report_lines.append(f"- `{artifacts}`")
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
                        f"- **SHAP Report:** {ml_artifacts.get('shap_report', 'N/A')}",
                        f"",
                        f"### Cross-Validation Metrics",
                        f"",
                        "```json",
                        json.dumps(ml_artifacts.get('metrics', {}), indent=2, default=str),
                        "```",
                        f"",
                    ])
            
            # Add SR detection method analysis
            elif step_name == 'sr_detection':
                # Analyze detection methods if SR levels are available
                sr_levels = step_result.get('sr_levels', [])
                if sr_levels:
                    method_analysis = self._analyze_sr_detection_methods(sr_levels)
                    if method_analysis:
                        report_lines.extend([
                            f"## Detection Method Analysis",
                            f"",
                            f"- **Total Levels Detected:** {method_analysis.get('total_levels', 0)}",
                            f"- **Detection Methods Used:** {method_analysis.get('total_methods', 0)}",
                            f"- **Most Effective Method:** {method_analysis.get('most_effective_method', 'unknown')}",
                            f"",
                            f"### Method Performance",
                            f"",
                        ])
                        
                        for method, stats in method_analysis.get('method_analysis', {}).items():
                            report_lines.extend([
                                f"#### {method.upper()}",
                                f"",
                                f"- **Total Levels:** {stats.get('total_levels', 0)}",
                                f"- **Support Levels:** {stats.get('support_levels', 0)}",
                                f"- **Resistance Levels:** {stats.get('resistance_levels', 0)}",
                                f"- **Average Strength:** {stats.get('avg_strength', 0.0):.4f}",
                                f"- **Strongest Level:** {stats.get('strongest_level', 0.0):.4f}",
                                f"- **Effectiveness Score:** {stats.get('effectiveness_score', 0.0):.2f}%",
                                f"- **Periods Used:** {stats.get('periods', [])}",
                                f"",
                            ])
                        
                        report_lines.append("")
            
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
                f"**Generated:** {dt.now().strftime('%Y-%m-%d %H:%M:%S')}",
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
                    # Handle both dict and list artifacts
                    if isinstance(artifacts, dict):
                        for key, value in artifacts.items():
                            report_lines.append(f"- **{key}:** `{value}`")
                    elif isinstance(artifacts, list):
                        for item in artifacts:
                            report_lines.append(f"- `{item}`")
                    else:
                        report_lines.append(f"- `{artifacts}`")
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
            # Step 0: Train ML model (enabled by default, skip only if --no-train-ml)
            if self.train_ml:
                self.logger.info("\n" + "=" * 80)
                self.logger.info("🧠 STEP 0: TRAIN SR QUALITY ML MODEL")
                self.logger.info("=" * 80)
                
                # Auto-set ML dates if not provided (use last 24 months of data for robustness)
                # CRITICAL FIX: Use NOW for ML training end date, NOT workflow end_date
                # ML training should use the maximum available data, independent of detection period
                if not self.ml_start_date or not self.ml_end_date:
                    # ALWAYS use current date for ML training (not workflow's end_date)
                    # This ensures we get the maximum amount of recent training data
                    end_dt = dt.now()
                    start_dt = end_dt - timedelta(days=730)  # 24 months training data
                    self.ml_start_date = start_dt.strftime('%Y-%m-%d')
                    self.ml_end_date = end_dt.strftime('%Y-%m-%d')
                    self.logger.info(f"   🔧 FIX: ML training uses latest available data (independent of workflow period)")
                    self.logger.info(f"   📅 ML training period: {self.ml_start_date} → {self.ml_end_date}")
                    self.logger.info(f"   ℹ️  This gives 24 months of training data ending NOW")
                
                self.logger.info(f"   ML Train Period: {self.ml_start_date} → {self.ml_end_date} | TF={self.ml_timeframe}")

                collector = SRQualityDataCollector()
                training_df = await collector.collect_training_data(
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

                # Add confidence weights (SOFT FILTERING / LABEL SMOOTHING)
                # Instead of discarding 75.6% garbage, weight samples by quality
                self.logger.info("\n🎯 Adding confidence weights to training data (label smoothing)...")
                training_df_weighted = collector.add_confidence_weights(
                    training_df,
                    method='tiered'  # Tiered weighting: noise=0.1x, strong=1.5x, critical=3.0x
                )
                
                model = SRQualityModel()
                
                # Use HPO to find optimal anti-overfitting configuration
                # WITH CONFIDENCE WEIGHTING ONLY (NO HARD FILTERING)
                self.logger.info("\n🎯 Training with Hyperparameter Optimization (HPO)...")
                self.logger.info("   🔬 Using GENTLE confidence weighting (preserve variance)")
                self.logger.info("      Noise gets 0.3x weight, Strong gets 1.2x, Critical gets 2.0x")
                self.logger.info("   📊 NO HARD FILTERING - keeping all data (weighted by quality)")
                self.logger.info("      Reason: Hard filtering caused model collapse (predicted ~0.81 for everything)")
                metrics = model.train_with_hpo(
                    training_df_weighted,
                    target_column='quality_score',
                    filter_percentile=100.0,  # NO FILTERING - use confidence weights only
                    n_trials=30,  # Optimized: reduced from 100 (diminishing returns after 30-40)
                    n_folds=3,  # Reduced from 5 to 3 for faster training
                    method='bayesian'  # Bayesian optimization for efficiency
                )
                
                model.save(self.ml_model_output)
                self.logger.info(f"✅ ML model saved to: {self.ml_model_output}")
                
                # Log HPO results
                if 'hpo_best_params' in metrics:
                    self.logger.info("🏆 Optimized Parameters:")
                    for param, value in metrics['hpo_best_params'].items():
                        self.logger.info(f"   {param}: {value}")
                self.logger.info(f"📊 ML CV avg Val R²: {metrics['avg_metrics']['avg_val_r2']:.4f}")
                
                # Evaluate RANKING metrics (what matters for SR detection!)
                self.logger.info("\n📊 Evaluating RANKING METRICS (what traders use)...")
                try:
                    # Use model's feature_names if available for proper feature selection
                    if model.feature_names is not None:
                        X_eval = training_df[model.feature_names]
                        self.logger.info(f"   Using {len(model.feature_names)} features from model")
                    else:
                        X_eval = training_df.filter(like='feature_')
                        self.logger.info(f"   Using {len(X_eval.columns)} features (filter by prefix)")
                    
                    y_eval = training_df['quality_score']
                    
                    ranking_metrics = model.evaluate_ranking(X_eval, y_eval, k=10)
                    
                    self.logger.info(f"   Precision@10:  {ranking_metrics['precision_at_k']*100:.1f}%")
                    self.logger.info(f"   Spearman ρ:    {ranking_metrics['spearman_rho']:.3f}")
                    self.logger.info(f"   NDCG@10:       {ranking_metrics['ndcg_at_k']:.3f}")
                    
                    # Add to metrics for reporting
                    metrics['ranking_metrics'] = ranking_metrics
                except Exception as e:
                    self.logger.warning(f"⚠️ Ranking evaluation failed: {e}")
                    import traceback
                    self.logger.debug(traceback.format_exc())
                
                # Generate SHAP explanations if enabled (without slow matplotlib plotting)
                shap_report_path = None
                if self.enable_shap_reporting:
                    self.logger.info("📊 Generating SHAP explanations (no plots for speed)...")
                    try:
                        import shap
                        import numpy as np

                        # Prepare data for SHAP: keep only numeric features
                        # Drop target and non-numeric columns
                        non_numeric_cols = ['quality_score', 'date', 'symbol', 'exchange', 'timeframe']

                        # Get numeric features only
                        numeric_df = training_df.select_dtypes(include=[np.number])
                        if 'quality_score' in numeric_df.columns:
                            numeric_df = numeric_df.drop(columns=['quality_score'])

                        self.logger.info(f"📊 SHAP analysis on {len(numeric_df.columns)} numeric features from {len(training_df)} samples")
                        
                        # Sample for SHAP (optimized: reduced from 1000 to 300 for faster computation)
                        shap_data = numeric_df.head(300)

                        # Get feature importance via SHAP (skip plotting for performance)
                        explainer = shap.TreeExplainer(model.model)
                        shap_values = explainer.shap_values(shap_data)

                        # Calculate feature importance without plotting
                        if isinstance(shap_values, list):
                            # Multi-class case
                            shap_values_mean = np.abs(shap_values[0]).mean(axis=0)
                        else:
                            # Binary/regression case
                            shap_values_mean = np.abs(shap_values).mean(axis=0)

                        # Get top features by importance
                        import pandas as pd
                        feature_importance = pd.DataFrame({
                            'feature': shap_data.columns,
                            'importance': shap_values_mean
                        }).sort_values('importance', ascending=False)

                        # Save feature importance to CSV instead of plots
                        shap_csv_path = self.outcomes_dir / f"shap_importance_{self.symbol}_{self.datetime_stamp}.csv"
                        feature_importance.to_csv(shap_csv_path, index=False)

                        self.logger.info(f"✅ SHAP feature importance saved: {shap_csv_path}")
                        self.logger.info(f"   Top 5 features: {', '.join(feature_importance.head(5)['feature'].tolist())}")

                        shap_report_path = str(shap_csv_path)
                        self.logger.info(f"✅ SHAP analysis complete - CSV saved (plots skipped for speed)")

                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to generate SHAP report: {e}")
                        import traceback
                        self.logger.debug(traceback.format_exc())
                
                ml_duration = (dt.now() - self.workflow_state['start_time']).total_seconds()
                workflow_results['steps_completed'].append('ml_model_training')
                workflow_results['artifacts']['ml_training'] = {
                    'training_data_path': saved_training_path,
                    'model_path': self.ml_model_output,
                    'metrics': metrics,
                    'shap_report': shap_report_path
                }
                
                # Generate ML training report with SHAP
                ml_report_result = {
                    'success': True,
                    'metrics': metrics,
                    'artifacts': {'ml_training': workflow_results['artifacts']['ml_training']},
                    'shap_enabled': self.enable_shap_reporting,
                    'shap_report_path': shap_report_path
                }
                ml_report_path = self._generate_step_report('ml_model_training', ml_report_result, ml_duration)
                self.report_paths['ml_model_training'] = ml_report_path
                
                # Step 0b: Validate ML Model Ranking Performance
                self.logger.info("\n" + "=" * 80)
                self.logger.info("🔬 STEP 0b: VALIDATE ML MODEL RANKING METRICS")
                self.logger.info("=" * 80)
                self.logger.info("   Testing if model actually ranks strong SR levels correctly...")
                
                try:
                    validation_start = dt.now()
                    validation_results = await validate_ranking_metrics(
                        symbol=self.symbol,
                        exchange=self.exchange,
                        timeframe=self.ml_timeframe,
                        ml_model_path=self.ml_model_output,
                        training_data_path=saved_training_path
                    )
                    
                    # Print detailed validation results
                    print_ranking_results(validation_results)
                    
                    # Check if model passed validation
                    passes = 0
                    total_tests = 0
                    
                    if 'precision_at_k' in validation_results:
                        for k, val in validation_results['precision_at_k'].items():
                            total_tests += 1
                            threshold = 0.80 if k <= 5 else 0.75
                            if val >= threshold:
                                passes += 1
                    
                    if 'spearman' in validation_results:
                        total_tests += 1
                        if validation_results['spearman'] >= 0.60:
                            passes += 1
                    
                    if 'separation' in validation_results:
                        total_tests += 1
                        if validation_results['separation']['separation'] >= 0.35:
                            passes += 1
                    
                    if 'future_generalization' in validation_results and validation_results['future_generalization']['r2'] is not None:
                        total_tests += 1
                        if validation_results['future_generalization']['r2'] >= 0.45:
                            passes += 1
                    
                    validation_success_rate = (passes / total_tests * 100) if total_tests > 0 else 0
                    
                    if passes == total_tests:
                        self.logger.info(f"✅ Model validation PASSED ({passes}/{total_tests} tests) - Production ready!")
                    elif passes >= total_tests * 0.75:
                        self.logger.info(f"⚠️  Model validation MOSTLY PASSED ({passes}/{total_tests} tests) - Minor issues")
                    elif passes >= total_tests * 0.5:
                        self.logger.info(f"⚠️  Model validation MARGINAL ({passes}/{total_tests} tests) - Needs improvement")
                    else:
                        self.logger.warning(f"❌ Model validation FAILED ({passes}/{total_tests} tests) - Significant issues detected")
                        self.logger.warning("   ⚠️  Continuing with workflow, but model may not perform well")
                    
                    validation_duration = (dt.now() - validation_start).total_seconds()
                    workflow_results['steps_completed'].append('ml_model_validation')
                    workflow_results['artifacts']['ml_validation'] = {
                        'validation_results': validation_results,
                        'tests_passed': passes,
                        'total_tests': total_tests,
                        'success_rate': validation_success_rate
                    }
                    workflow_results['metrics']['ml_validation'] = {
                        'tests_passed': passes,
                        'total_tests': total_tests,
                        'success_rate': validation_success_rate,
                        'precision_at_10': validation_results.get('precision_at_k', {}).get(10, None),
                        'spearman_rho': validation_results.get('spearman', None),
                        'separation': validation_results.get('separation', {}).get('separation', None)
                    }
                    
                    # Generate validation report
                    validation_report_result = {
                        'success': True,
                        'metrics': workflow_results['metrics']['ml_validation'],
                        'artifacts': {'ml_validation': workflow_results['artifacts']['ml_validation']},
                        'validation_results': validation_results
                    }
                    validation_report_path = self._generate_step_report('ml_model_validation', validation_report_result, validation_duration)
                    self.report_paths['ml_model_validation'] = validation_report_path
                    
                except Exception as e:
                    self.logger.warning(f"⚠️  Model validation failed: {e}")
                    import traceback
                    self.logger.debug(traceback.format_exc())
                    self.logger.warning("   Continuing with workflow despite validation failure...")
                    workflow_results['steps_failed'].append('ml_model_validation')
                    workflow_results['errors']['ml_model_validation'] = str(e)

            # Step 1: SR Parameter Optimization
            self.logger.info("\n" + "=" * 80)
            self.logger.info("📊 STEP 1: SR PARAMETER OPTIMIZATION (ENHANCED)")
            if self.enhanced_sr_config:
                self.logger.info(f"   ⚡ Using enhanced configuration:")
                self.logger.info(f"      - Trials: {self.enhanced_sr_config.n_trials}")
                self.logger.info(f"      - Coarse grid points: {self.enhanced_sr_config.coarse_grid_points}")
                self.logger.info(f"      - Fine grid points: {self.enhanced_sr_config.fine_grid_points}")
                self.logger.info(f"      - TPE trials: {self.enhanced_sr_config.tpe_trials}")
                self.logger.info(f"      - Optimization level: {self.enhanced_sr_config.optimization_level}")
                self.logger.info(f"      - Max workers: {self.enhanced_sr_config.max_workers}")
            else:
                self.logger.info("   ⚠️ Using default configuration (enhanced config not loaded)")
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
            
            # Execute optimization with enhanced configuration
            if self.enhanced_sr_config:
                self.logger.info("🚀 Executing SR parameter optimization with enhanced configuration...")
                optimization_result = await self.param_optimizer.execute(
                    optimization_config,
                    enhanced_config=self.enhanced_sr_config
                )
            else:
                self.logger.info("🚀 Executing SR parameter optimization with default configuration...")
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
            opt_duration = (dt.now() - self.workflow_state['start_time']).total_seconds()
            opt_report_path = self._generate_step_report('sr_parameter_optimization', optimization_result, opt_duration)
            self.report_paths['sr_parameter_optimization'] = opt_report_path
            
            # Extract optimized parameters from artifacts (not metrics)
            optimized_params = self._extract_optimized_parameters(optimization_result)
            if optimized_params:
                self.logger.info(f"📊 Loaded {len(optimized_params)} optimized parameters")
            else:
                self.logger.warning("⚠️ No optimized parameters found, detection will use defaults")
            
            # Step 2: SR Detection with optimized parameters using EnhancedSRDetector with ML
            self.logger.info("\n" + "=" * 80)
            self.logger.info("📊 STEP 2: SR DETECTION WITH ML MODEL AND OPTIMIZED PARAMETERS")
            self.logger.info("=" * 80)
            
            # Apply optimized parameters to detector with proper name mapping
            if optimized_params:
                # Map parameter names from optimization to detector attribute names
                param_mapping = {
                    'min_touches': 'min_touches',
                    'strength_threshold': 'min_strength',
                    'distance_threshold': 'touch_proximity_threshold',
                    'lookback_periods': None,  # Skip - detector doesn't use this parameter
                    'volume_threshold': 'volume_spike_threshold',
                }
                
                # Override optimized min_touches to prioritize coverage over extreme selectivity
                if 'min_touches' in optimized_params:
                    self.logger.info(f"   🔧 Overriding optimized min_touches ({optimized_params['min_touches']}) with 2 for better coverage")
                    optimized_params['min_touches'] = 2
                
                applied_count = 0
                for param_name, param_value in optimized_params.items():
                    # Map to detector attribute name
                    attr_name = param_mapping.get(param_name, param_name)
                    
                    # Skip if explicitly set to None
                    if attr_name is None:
                        self.logger.info(f"   ⊘ Skipped parameter: {param_name} (not used by detector)")
                        continue
                    
                    if hasattr(self.sr_detector, attr_name):
                        setattr(self.sr_detector, attr_name, param_value)
                        self.logger.info(f"   ✓ Applied optimized parameter: {param_name} ({attr_name}) = {param_value}")
                        applied_count += 1
                    else:
                        self.logger.warning(f"   ⚠ Detector has no attribute '{attr_name}' for parameter '{param_name}'")
                
                if applied_count > 0:
                    self.logger.info(f"✅ Applied {applied_count}/{len(optimized_params)} optimized parameters to detector")
                else:
                    self.logger.warning(f"⚠️ Could not apply any optimized parameters - detector may not support them")
            
            # CRITICAL: Re-apply clustering disable flags (they may have been overridden)
            self.sr_detector.disable_dbscan_clustering = True
            self.sr_detector.config['disable_backtesting_validation'] = True
            self.logger.info("✅ Clustering disabled flags applied to detector")
            
            # Fetch market data for detection
            self.logger.info(f"📊 Fetching market data for {self.symbol}...")
            try:
                market_data = await self.data_loader.load_market_data(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    timeframe=self.timeframe,
                    start_date=self.start_date,
                    end_date=self.end_date
                )
                
                if market_data is None or market_data.empty:
                    error_msg = "Failed to fetch market data for SR detection"
                    self.logger.error(f"❌ {error_msg}")
                    workflow_results['steps_failed'].append('sr_detection')
                    workflow_results['errors']['sr_detection'] = error_msg
                    return workflow_results
                
                self.logger.info(f"✅ Fetched {len(market_data)} data points")
                
                # Perform ML-based SR detection
                self.logger.info("🧠 Running ML-based SR detection...")
                detected_levels = self.sr_detector.detect_sr_levels(market_data)
                
                # Convert SRLevel objects to dict format for compatibility
                # Handle both 'type' and 'level_type' attribute names
                support_levels = [level for level in detected_levels 
                                 if (hasattr(level, 'type') and level.type == 'support') or 
                                    (hasattr(level, 'level_type') and level.level_type == 'support')]
                resistance_levels = [level for level in detected_levels 
                                    if (hasattr(level, 'type') and level.type == 'resistance') or 
                                       (hasattr(level, 'level_type') and level.level_type == 'resistance')]
                
                # Optimized: Use to_dict() method for efficient conversion
                detection_result = {
                    'total_levels': len(detected_levels),
                    'support_levels': len(support_levels),
                    'resistance_levels': len(resistance_levels),
                    'levels': [level.to_dict() if hasattr(level, 'to_dict') else {
                        'price': level.price,
                        'type': level.type if hasattr(level, 'type') else (level.level_type if hasattr(level, 'level_type') else 'unknown'),
                        'strength': level.strength,
                        'touches': level.touches if hasattr(level, 'touches') else (level.touch_count if hasattr(level, 'touch_count') else 1),
                        'method': level.method if hasattr(level, 'method') else 'unknown',
                        'quality_score': level.quality_score if hasattr(level, 'quality_score') else level.strength
                    } for level in detected_levels],
                    'metadata': {
                        'ml_model_used': True,
                        'optimized_parameters_applied': (applied_count > 0) if optimized_params else False,
                        'parameters_applied_count': applied_count if optimized_params else 0,
                        'total_parameters_available': len(optimized_params) if optimized_params else 0
                    }
                }
                
                self.logger.info("✅ SR detection completed successfully")
                workflow_results['steps_completed'].append('sr_detection')
                
                # Save detection result as artifact
                artifact_path = self.outcomes_dir / f"sr_detection_{self.symbol}_{self.timeframe}_{self.datetime_stamp}.json"
                with open(artifact_path, 'w') as f:
                    json.dump(detection_result, f, indent=2, default=str)
                
                workflow_results['artifacts']['detection'] = {'sr_detection_result': str(artifact_path)}
                workflow_results['metrics']['detection'] = {
                    'total_levels': detection_result['total_levels'],
                    'support_levels': detection_result['support_levels'],
                    'resistance_levels': detection_result['resistance_levels'],
                    'ml_model_used': True
                }
                
                # Generate SR detection report
                det_start = dt.now()
                det_duration = (det_start - self.workflow_state['start_time']).total_seconds()
                det_report_result = {
                    'success': True,
                    'sr_levels': detection_result.get('levels', []),  # Include levels for method analysis
                    'metrics': workflow_results['metrics']['detection'],
                    'artifacts': workflow_results['artifacts']['detection']
                }
                det_report_path = self._generate_step_report('sr_detection', det_report_result, det_duration)
                self.report_paths['sr_detection'] = det_report_path
                
                total_levels = detection_result['total_levels']
                self.logger.info(f"📊 Detected {total_levels} SR levels using ML model")
                
                # Store sr_levels for clustering
                sr_levels = detection_result
                
            except Exception as e:
                error_msg = f"SR detection failed: {str(e)}"
                self.logger.error(f"❌ {error_msg}")
                import traceback
                self.logger.error(traceback.format_exc())
                workflow_results['steps_failed'].append('sr_detection')
                workflow_results['errors']['sr_detection'] = error_msg
                return workflow_results
            
            # Step 3: Filter Weak SR Levels (replaces clustering step)
            self.logger.info("\n" + "=" * 80)
            self.logger.info("🎯 STEP 3: FILTER WEAK SR LEVELS")
            self.logger.info("=" * 80)
            self.logger.info("   Using ML model/strength scores to keep only high-quality levels")
            self.logger.info("   Clustering step removed - ML model handles level selection")
            
            # Filter weak levels based on strength threshold
            min_strength_threshold = 0.4  # Keep levels with strength >= 0.4 (lowered for better coverage)
            sr_levels_list = detection_result.get('levels', [])
            
            filtered_levels = []
            weak_levels_removed = 0
            
            for level in sr_levels_list:
                strength = getattr(level, 'strength', None) if hasattr(level, 'strength') else level.get('strength', 0)
                if strength >= min_strength_threshold:
                    filtered_levels.append(level)
                else:
                    weak_levels_removed += 1
            
            self.logger.info(f"📊 Filtered levels: {len(sr_levels_list)} → {len(filtered_levels)} (removed {weak_levels_removed} weak levels)")
            self.logger.info(f"   Strength threshold: {min_strength_threshold}")
            self.logger.info(f"   Support levels: {len([l for l in filtered_levels if (getattr(l, 'type', None) if hasattr(l, 'type') else l.get('type')) == 'support'])}")
            self.logger.info(f"   Resistance levels: {len([l for l in filtered_levels if (getattr(l, 'type', None) if hasattr(l, 'type') else l.get('type')) == 'resistance'])}")
            
            workflow_results['steps_completed'].append('sr_filtering')
            workflow_results['artifacts']['filtering'] = {
                'filtered_sr_levels': len(filtered_levels),
                'removed_weak_levels': weak_levels_removed,
                'strength_threshold': min_strength_threshold
            }
            workflow_results['metrics']['filtering'] = {
                'total_levels_before': len(sr_levels_list),
                'total_levels_after': len(filtered_levels),
                'weak_levels_removed': weak_levels_removed,
                'retention_rate': len(filtered_levels) / len(sr_levels_list) if sr_levels_list else 0
            }
            
            # Mark workflow as successful
            workflow_results['success'] = True
            workflow_results['end_time'] = dt.now()
            workflow_results['total_duration'] = (
                workflow_results['end_time'] - self.workflow_state['start_time']
            ).total_seconds()
            
            self.logger.info("\n" + "=" * 80)
            self.logger.info("✅ SR WORKFLOW COMPLETED SUCCESSFULLY")
            # Calculate totals for summary
            total_steps = 5 if self.train_ml else 3  # ML Training + Validation + Optimization + Detection + Filtering OR just Optimization + Detection + Filtering
            param_count = len(optimized_params) if isinstance(optimized_params, dict) else 0
            self.logger.info(f"   Steps completed: {len(workflow_results['steps_completed'])}/{total_steps}")
            self.logger.info(f"   Total duration: {workflow_results['total_duration']:.2f}s")
            self.logger.info(f"   Optimized parameters: {param_count} params")
            self.logger.info(f"   Detected SR levels: {total_levels}")
            self.logger.info(f"   High-quality levels (filtered): {len(filtered_levels)}")
            self.logger.info(f"   Weak levels removed: {weak_levels_removed}")
            
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
            workflow_results['end_time'] = dt.now()
            return workflow_results


async def main():
    """Main entry point for the SR workflow runner."""
    parser = argparse.ArgumentParser(
        description="Run the complete SR workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (ML training + SHAP enabled by default)
  python scripts/run_sr_workflow.py --symbol ETHUSDT --exchange binance --timeframe 15m
  
  # With 30-day lookback (still includes ML training with auto-set dates)
  python scripts/run_sr_workflow.py --symbol BTCUSDT --exchange binance --timeframe 1h --lookback-days 30
  
  # Skip ML training if model already exists
  python scripts/run_sr_workflow.py --symbol ETHUSDT --timeframe 15m --no-train-ml
  
  # Custom ML training dates
  python scripts/run_sr_workflow.py --symbol ETHUSDT --timeframe 15m --ml-start-date 2025-06-01 --ml-end-date 2025-10-31
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
    
    # ML training stage (ENABLED BY DEFAULT - trains SR quality prediction model)
    parser.add_argument('--no-train-ml', action='store_true', 
                       help='Skip ML training stage (default: ENABLED - trains LGBM model for SR quality scoring)')
    parser.add_argument('--ml-start-date', type=str, default=None, 
                       help='ML training start date (YYYY-MM-DD). Auto-set to 6 months ago if not provided')
    parser.add_argument('--ml-end-date', type=str, default=None, 
                       help='ML training end date (YYYY-MM-DD). Auto-set to today if not provided')
    parser.add_argument('--ml-timeframe', type=str, default=None, 
                       help='ML training timeframe (defaults to main --timeframe)')
    parser.add_argument('--ml-model-output', type=str, default='models/sr_quality_model.lgb', 
                       help='Path to save trained ML model')
    parser.add_argument('--ml-sample-freq-days', type=float, default=0.5, 
                       help='Sampling frequency in days for ML training data collection (default: 0.5 = 12-hour for more samples)')
    parser.add_argument('--ml-forward-days', type=int, default=10, 
                       help='Forward window (days) to measure SR performance for labeling (default: 10)')
    parser.add_argument('--no-shap', action='store_true', 
                       help='Disable SHAP feature importance plots (default: ENABLED - generates SHAP summary plots)')
    
    args = parser.parse_args()
    
    # Validate date arguments
    if args.end_date:
        try:
            dt.strptime(args.end_date, '%Y-%m-%d')
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
        train_ml=not args.no_train_ml,  # ML training enabled by default
        ml_start_date=args.ml_start_date,
        ml_end_date=args.ml_end_date,
        ml_timeframe=args.ml_timeframe,
        ml_model_output=args.ml_model_output,
        ml_sample_freq_days=args.ml_sample_freq_days,
        ml_forward_days=args.ml_forward_days,
        enable_shap_reporting=not args.no_shap  # SHAP enabled by default
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
        print("\nResults:")
        print(f"  Optimized parameters: {len(results.get('metrics', {}).get('optimization', {}))} params")
        print(f"  Detected SR levels: {results.get('metrics', {}).get('detection', {}).get('total_levels', 0)}")
        filtering_metrics = results.get('metrics', {}).get('filtering', {})
        if filtering_metrics:
            print(f"  High-quality levels: {filtering_metrics.get('total_levels_after', 0)}")
            print(f"  Weak levels removed: {filtering_metrics.get('weak_levels_removed', 0)}")
            print(f"  Retention rate: {filtering_metrics.get('retention_rate', 0)*100:.1f}%")
        print("\nArtifacts created:")
        for step_name, artifacts in results.get('artifacts', {}).items():
            if isinstance(artifacts, dict):
                print(f"  {step_name}: {len(artifacts)} artifacts")
            else:
                print(f"  {step_name}: 1 artifact")
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
