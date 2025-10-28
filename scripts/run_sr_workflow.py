#!/usr/bin/env python3
"""
SR Workflow Runner

This script runs the complete SR (Support/Resistance) workflow in the correct order:
1. SR Parameter Optimization - Optimizes parameters for SR detection
2. SR Detection - Detects SR levels using optimized parameters
3. SR Clustering - Clusters the detected SR levels

Usage:
    python scripts/run_sr_workflow.py --symbol ETHUSDT --exchange binance --timeframe 15m --mode light
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.training.steps.base_step import step_registry

# Import SR workflow steps
from src.training.steps.market_analysis.components.sr_parameter_optimization import SRParameterOptimizationStep
from src.training.steps.market_analysis.components.sr_detection import SRDetectionComponent
from src.training.steps.market_analysis.components.sr_clustering import SRClusteringComponent


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
        direction: str = "longs",
        mode: str = "light"
    ):
        """
        Initialize the SR workflow runner.
        
        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            exchange: Exchange name (e.g., 'binance')
            timeframe: Timeframe (e.g., '15m')
            direction: Trading direction ('longs' or 'shorts')
            mode: Execution mode ('light', 'full', or 'blank')
        """
        self.logger = system_logger.getChild('SRWorkflowRunner')
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe
        self.direction = direction
        self.mode = mode
        
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
        
        self.logger.info("✅ SR workflow runner initialized")
    
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
                'enable_sr_detection_testing': True  # Enable SR detection testing during optimization
            }
            
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
            
            # Extract optimized parameters
            optimized_params = optimization_result.get('metrics', {}).get('optimized_parameters', {})
            self.logger.info(f"📊 Optimized parameters: {optimized_params}")
            
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
            
            # Extract detected SR levels
            sr_levels = detection_result.get('detection_result', {})
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
            
            # Extract clustering results
            clusters = clustering_result.get('clustering_result', {})
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
            self.logger.info(f"   Steps completed: {len(workflow_results['steps_completed'])}/3")
            self.logger.info(f"   Total duration: {workflow_results['total_duration']:.2f}s")
            self.logger.info(f"   Optimized parameters: {len(optimized_params)} params")
            self.logger.info(f"   Detected SR levels: {total_levels}")
            self.logger.info(f"   Created clusters: {total_clusters}")
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
    parser = argparse.ArgumentParser(description="Run the complete SR workflow")
    parser.add_argument('--symbol', type=str, default='ETHUSDT', help='Trading symbol')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange name')
    parser.add_argument('--timeframe', type=str, default='15m', help='Timeframe')
    parser.add_argument('--direction', type=str, default='longs', choices=['longs', 'shorts'], help='Trading direction')
    parser.add_argument('--mode', type=str, default='light', choices=['light', 'full', 'blank'], help='Execution mode')
    
    args = parser.parse_args()
    
    # Create workflow runner
    runner = SRWorkflowRunner(
        symbol=args.symbol,
        exchange=args.exchange,
        timeframe=args.timeframe,
        direction=args.direction,
        mode=args.mode
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
