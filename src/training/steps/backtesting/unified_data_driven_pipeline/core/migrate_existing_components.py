"""
Migration Script for Existing Backtesting Components

This script provides automated migration of existing backtesting components
to use the ModularComponent architecture. It includes analysis, migration,
and validation capabilities.

Usage:
    python migrate_existing_components.py --analyze
    python migrate_existing_components.py --migrate --component real_monte_carlo_engine
    python migrate_existing_components.py --migrate-all
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import time

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent.parent))

from src.training.steps.backtesting.unified_data_driven_pipeline.core.migration_utils import (
    BacktestingComponentAnalyzer,
    BacktestingComponentMigrator,
    create_backtesting_component_wrapper,
    validate_backtesting_migration_compatibility,
    generate_backtesting_migration_report
)
from src.training.steps.backtesting.unified_data_driven_pipeline.core.component_registry import (
    get_registry,
    register_component,
    ComponentType
)
from src.training.steps.backtesting.unified_data_driven_pipeline.core.modular_architecture import (
    ModularComponent,
    create_backtesting_component
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BacktestingComponentMigrationManager:
    """Manager for migrating existing backtesting components."""
    
    def __init__(self):
        self.analyzer = BacktestingComponentAnalyzer()
        self.migrator = BacktestingComponentMigrator()
        self.registry = get_registry()
        
        # Define components to migrate
        self.components_to_migrate = {
            # Core backtesting components
            'real_monte_carlo_engine': {
                'file': 'src/training/steps/backtesting/real_monte_carlo_engine.py',
                'class': 'RealMonteCarloEngine',
                'type': ComponentType.MONTE_CARLO_ENGINE,
                'dependencies': ['data_loader', 'feature_generator'],
                'priority': 'high'
            },
            'real_parameters_optimization': {
                'file': 'src/training/steps/backtesting/real_parameters_optimization.py',
                'class': 'RealParametersOptimizer',
                'type': ComponentType.PARAMETER_OPTIMIZER,
                'dependencies': ['data_loader', 'feature_generator'],
                'priority': 'high'
            },
            'real_reporting_engine': {
                'file': 'src/training/steps/backtesting/real_reporting_engine.py',
                'class': 'RealReportingEngine',
                'type': ComponentType.REPORTING_ENGINE,
                'dependencies': ['backtesting_engine', 'performance_analyzer'],
                'priority': 'high'
            },
            'vectorbt_unified_manager': {
                'file': 'src/training/steps/backtesting/vectorbt_unified_manager.py',
                'class': 'VectorBTUnifiedManager',
                'type': ComponentType.VECTORBT_MANAGER,
                'dependencies': ['data_loader'],
                'priority': 'high'
            },
            'final_parameters_optimization': {
                'file': 'src/training/steps/backtesting/final_parameters_optimization.py',
                'class': 'FinalParametersOptimizer',
                'type': ComponentType.PARAMETER_OPTIMIZER,
                'dependencies': ['data_loader', 'feature_generator'],
                'priority': 'medium'
            },
            
            # ABC Testing components
            'paper_trading_engine': {
                'file': 'src/training/steps/backtesting/abc_testing/paper_trading_engine.py',
                'class': 'PaperTradingEngine',
                'type': ComponentType.PAPER_TRADING_ENGINE,
                'dependencies': ['data_loader', 'risk_management'],
                'priority': 'high'
            },
            'performance_monitoring': {
                'file': 'src/training/steps/backtesting/abc_testing/performance_monitoring.py',
                'class': 'PerformanceMonitor',
                'type': ComponentType.PERFORMANCE_MONITOR,
                'dependencies': ['backtesting_engine'],
                'priority': 'medium'
            },
            'risk_management': {
                'file': 'src/training/steps/backtesting/abc_testing/risk_management.py',
                'class': 'RiskManager',
                'type': ComponentType.RISK_MANAGER,
                'dependencies': ['data_loader'],
                'priority': 'high'
            },
            'statistical_analysis': {
                'file': 'src/training/steps/backtesting/abc_testing/statistical_analysis.py',
                'class': 'StatisticalAnalyzer',
                'type': ComponentType.STATISTICAL_ANALYZER,
                'dependencies': ['data_loader'],
                'priority': 'medium'
            },
            
            # NAS TAS components
            'walk_forward_analyzer': {
                'file': 'src/training/steps/backtesting/nas_tas_deprecated/walk_forward_analyzer.py',
                'class': 'WalkForwardAnalyzer',
                'type': ComponentType.WALK_FORWARD_ANALYZER,
                'dependencies': ['data_loader', 'backtesting_engine'],
                'priority': 'medium'
            },
            'performance_attribution': {
                'file': 'src/training/steps/backtesting/nas_tas_deprecated/performance_attribution.py',
                'class': 'PerformanceAttribution',
                'type': ComponentType.PERFORMANCE_ATTRIBUTION,
                'dependencies': ['backtesting_engine', 'performance_analyzer'],
                'priority': 'low'
            }
        }
    
    def analyze_component(self, component_name: str) -> Dict[str, Any]:
        """Analyze a specific component."""
        if component_name not in self.components_to_migrate:
            raise ValueError(f"Component {component_name} not found in migration list")
        
        component_info = self.components_to_migrate[component_name]
        file_path = component_info['file']
        class_name = component_info['class']
        
        logger.info(f"Analyzing component: {component_name}")
        logger.info(f"File: {file_path}")
        logger.info(f"Class: {class_name}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {
                'component_name': component_name,
                'status': 'error',
                'error': f"File not found: {file_path}"
            }
        
        try:
            # Analyze the component
            analysis = self.analyzer.analyze_component(file_path, class_name)
            
            result = {
                'component_name': component_name,
                'file_path': file_path,
                'class_name': class_name,
                'status': 'analyzed',
                'analysis': analysis.__dict__,
                'migration_recommendation': self._get_migration_recommendation(analysis)
            }
            
            logger.info(f"Analysis completed for {component_name}")
            logger.info(f"Compatibility score: {analysis.compatibility_score:.2f}")
            logger.info(f"Migration complexity: {analysis.migration_complexity}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing component {component_name}: {e}")
            return {
                'component_name': component_name,
                'status': 'error',
                'error': str(e)
            }
    
    def analyze_all_components(self) -> Dict[str, Any]:
        """Analyze all components."""
        logger.info("Analyzing all components...")
        
        results = {}
        for component_name in self.components_to_migrate:
            try:
                results[component_name] = self.analyze_component(component_name)
            except Exception as e:
                logger.error(f"Error analyzing {component_name}: {e}")
                results[component_name] = {
                    'component_name': component_name,
                    'status': 'error',
                    'error': str(e)
                }
        
        # Generate summary
        total_components = len(self.components_to_migrate)
        analyzed_components = sum(1 for r in results.values() if r['status'] == 'analyzed')
        error_components = sum(1 for r in results.values() if r['status'] == 'error')
        
        summary = {
            'total_components': total_components,
            'analyzed_components': analyzed_components,
            'error_components': error_components,
            'analysis_timestamp': time.time(),
            'results': results
        }
        
        logger.info(f"Analysis completed: {analyzed_components}/{total_components} components analyzed")
        
        return summary
    
    def migrate_component(self, component_name: str, strategy: Optional[str] = None) -> Dict[str, Any]:
        """Migrate a specific component."""
        if component_name not in self.components_to_migrate:
            raise ValueError(f"Component {component_name} not found in migration list")
        
        component_info = self.components_to_migrate[component_name]
        file_path = component_info['file']
        class_name = component_info['class']
        
        logger.info(f"Migrating component: {component_name}")
        logger.info(f"File: {file_path}")
        logger.info(f"Class: {class_name}")
        
        try:
            # First analyze the component
            analysis = self.analyzer.analyze_component(file_path, class_name)
            
            # Determine migration strategy
            if strategy is None:
                strategy = self._get_migration_recommendation(analysis)
            
            logger.info(f"Using migration strategy: {strategy}")
            
            # Migrate the component
            result = self.migrator.migrate_component(file_path, class_name, strategy)
            
            if result.success:
                logger.info(f"Migration successful for {component_name}")
                
                # Register the migrated component
                self._register_migrated_component(component_name, result, component_info)
                
                return {
                    'component_name': component_name,
                    'status': 'migrated',
                    'strategy': strategy,
                    'migration_result': result.__dict__,
                    'registered': True
                }
            else:
                logger.error(f"Migration failed for {component_name}: {result.issues}")
                return {
                    'component_name': component_name,
                    'status': 'failed',
                    'strategy': strategy,
                    'migration_result': result.__dict__,
                    'error': result.issues
                }
                
        except Exception as e:
            logger.error(f"Error migrating component {component_name}: {e}")
            return {
                'component_name': component_name,
                'status': 'error',
                'error': str(e)
            }
    
    def migrate_all_components(self, strategy: Optional[str] = None) -> Dict[str, Any]:
        """Migrate all components."""
        logger.info("Migrating all components...")
        
        results = {}
        successful_migrations = 0
        failed_migrations = 0
        
        # Sort components by priority
        sorted_components = sorted(
            self.components_to_migrate.items(),
            key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x[1]['priority']]
        )
        
        for component_name, component_info in sorted_components:
            try:
                logger.info(f"Migrating component: {component_name} (priority: {component_info['priority']})")
                result = self.migrate_component(component_name, strategy)
                results[component_name] = result
                
                if result['status'] == 'migrated':
                    successful_migrations += 1
                else:
                    failed_migrations += 1
                    
            except Exception as e:
                logger.error(f"Error migrating {component_name}: {e}")
                results[component_name] = {
                    'component_name': component_name,
                    'status': 'error',
                    'error': str(e)
                }
                failed_migrations += 1
        
        summary = {
            'total_components': len(self.components_to_migrate),
            'successful_migrations': successful_migrations,
            'failed_migrations': failed_migrations,
            'migration_timestamp': time.time(),
            'results': results
        }
        
        logger.info(f"Migration completed: {successful_migrations}/{len(self.components_to_migrate)} components migrated successfully")
        
        return summary
    
    def _get_migration_recommendation(self, analysis) -> str:
        """Get migration recommendation based on analysis."""
        if analysis.compatibility_score >= 0.8:
            return 'direct'
        elif analysis.compatibility_score >= 0.6:
            return 'refactor'
        elif analysis.compatibility_score >= 0.4:
            return 'wrapper'
        else:
            return 'rewrite'
    
    def _register_migrated_component(self, component_name: str, migration_result, component_info: Dict[str, Any]) -> None:
        """Register the migrated component in the registry."""
        try:
            # Get the migrated component class
            migrated_component = migration_result.migrated_component
            
            # Register in the registry
            self.registry.register_component(
                name=component_name,
                component_type=component_info['type'],
                component_class=migrated_component,
                dependencies=component_info['dependencies'],
                metadata={
                    'migrated': True,
                    'original_file': component_info['file'],
                    'migration_strategy': migration_result.strategy,
                    'migration_timestamp': time.time()
                }
            )
            
            logger.info(f"Component {component_name} registered in registry")
            
        except Exception as e:
            logger.error(f"Error registering component {component_name}: {e}")
    
    def create_wrapper_component(self, component_name: str) -> Dict[str, Any]:
        """Create a wrapper component for backward compatibility."""
        if component_name not in self.components_to_migrate:
            raise ValueError(f"Component {component_name} not found in migration list")
        
        component_info = self.components_to_migrate[component_name]
        file_path = component_info['file']
        class_name = component_info['class']
        
        try:
            # Import the original component
            import importlib.util
            spec = importlib.util.spec_from_file_location(class_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            original_class = getattr(module, class_name)
            
            # Create wrapper
            wrapper_class = create_backtesting_component_wrapper(original_class)
            
            # Register wrapper
            self.registry.register_component(
                name=f"{component_name}_wrapper",
                component_type=component_info['type'],
                component_class=wrapper_class,
                dependencies=component_info['dependencies'],
                metadata={
                    'wrapper': True,
                    'original_file': file_path,
                    'original_class': class_name,
                    'creation_timestamp': time.time()
                }
            )
            
            logger.info(f"Wrapper created for component {component_name}")
            
            return {
                'component_name': component_name,
                'status': 'wrapped',
                'wrapper_class': wrapper_class,
                'registered': True
            }
            
        except Exception as e:
            logger.error(f"Error creating wrapper for {component_name}: {e}")
            return {
                'component_name': component_name,
                'status': 'error',
                'error': str(e)
            }
    
    def generate_migration_report(self, results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Generate a comprehensive migration report."""
        if output_file is None:
            output_file = f"migration_report_{int(time.time())}.json"
        
        # Generate the report
        report = generate_backtesting_migration_report(results)
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Migration report saved to: {output_file}")
        return output_file


def main():
    """Main function for the migration script."""
    parser = argparse.ArgumentParser(description='Migrate existing backtesting components to ModularComponent')
    parser.add_argument('--analyze', action='store_true', help='Analyze all components')
    parser.add_argument('--analyze-component', type=str, help='Analyze a specific component')
    parser.add_argument('--migrate', type=str, help='Migrate a specific component')
    parser.add_argument('--migrate-all', action='store_true', help='Migrate all components')
    parser.add_argument('--create-wrapper', type=str, help='Create wrapper for a specific component')
    parser.add_argument('--strategy', type=str, choices=['direct', 'wrapper', 'refactor', 'rewrite'], 
                       help='Migration strategy to use')
    parser.add_argument('--output', type=str, help='Output file for reports')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    manager = BacktestingComponentMigrationManager()
    
    try:
        if args.analyze:
            results = manager.analyze_all_components()
            output_file = manager.generate_migration_report(results, args.output)
            print(f"Analysis completed. Report saved to: {output_file}")
            
        elif args.analyze_component:
            result = manager.analyze_component(args.analyze_component)
            print(json.dumps(result, indent=2, default=str))
            
        elif args.migrate:
            result = manager.migrate_component(args.migrate, args.strategy)
            print(json.dumps(result, indent=2, default=str))
            
        elif args.migrate_all:
            results = manager.migrate_all_components(args.strategy)
            output_file = manager.generate_migration_report(results, args.output)
            print(f"Migration completed. Report saved to: {output_file}")
            
        elif args.create_wrapper:
            result = manager.create_wrapper_component(args.create_wrapper)
            print(json.dumps(result, indent=2, default=str))
            
        else:
            parser.print_help()
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()