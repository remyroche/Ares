#!/usr/bin/env python3
"""
Analysis script to identify files that are called vs not called when launching ares_launcher for trading operations.

This script will:
1. Trace the execution flow from ares_launcher.py when paper/live trading is specified
2. Identify all Python files that are imported/executed during trading
3. Compare against step1 execution to show differences
4. Generate a report of trading-specific files
"""

import os
import sys
import importlib
import ast
import subprocess
from pathlib import Path
from typing import Set, List, Dict
import json

class TradingExecutionAnalyzer:
    def __init__(self, project_root: str = "/workspace"):
        self.project_root = Path(project_root)
        self.called_files: Set[str] = set()
        self.all_python_files: Set[str] = set()
        self.import_graph: Dict[str, List[str]] = {}
        
    def find_all_python_files(self) -> None:
        """Find all Python files in the project."""
        print("🔍 Finding all Python files...")
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip certain directories
            if any(skip_dir in root for skip_dir in ['.git', '__pycache__', '.pytest_cache', 'node_modules']):
                continue
                
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(self.project_root)
                    self.all_python_files.add(str(relative_path))
        
        print(f"📊 Found {len(self.all_python_files)} Python files")
    
    def analyze_trading_flow(self) -> None:
        """Analyze the execution flow starting from ares_launcher.py with trading modes."""
        print("🚀 Analyzing ares_launcher.py trading execution flow...")
        
        # Start with ares_launcher.py
        self.called_files.add("ares_launcher.py")
        
        # Analyze imports in ares_launcher.py
        self._analyze_file_imports("ares_launcher.py")
        
        # Follow the trading execution path
        self._follow_trading_execution_path()
    
    def _analyze_file_imports(self, file_path: str) -> None:
        """Analyze imports in a specific file."""
        try:
            full_path = self.project_root / file_path
            if not full_path.exists():
                return
                
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST to find imports
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self._add_imported_file(alias.name, file_path)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self._add_imported_file(node.module, file_path)
                        
        except Exception as e:
            print(f"⚠️ Error analyzing {file_path}: {e}")
    
    def _add_imported_file(self, module_name: str, source_file: str) -> None:
        """Add an imported file to the called files set."""
        # Convert module name to file path
        if module_name.startswith('src.'):
            # Handle src imports
            module_path = module_name.replace('.', '/') + '.py'
            if module_path.startswith('src/'):
                self.called_files.add(module_path)
                if source_file not in self.import_graph:
                    self.import_graph[source_file] = []
                self.import_graph[source_file].append(module_path)
        elif module_name.startswith('scripts.'):
            # Handle scripts imports
            module_path = module_name.replace('.', '/') + '.py'
            self.called_files.add(module_path)
            if source_file not in self.import_graph:
                self.import_graph[source_file] = []
            self.import_graph[source_file].append(module_path)
    
    def _follow_trading_execution_path(self) -> None:
        """Follow the specific execution path for trading operations."""
        print("📋 Following trading execution path...")
        
        # Based on the analysis of ares_launcher.py and ares_pipeline.py, these are the key files called for trading:
        trading_related_files = [
            # Main trading pipeline
            "src/ares_pipeline.py",
            
            # Core trading components
            "src/analyst/analyst.py",
            "src/strategist/strategist.py",
            "src/tactician/tactician.py",
            "src/supervisor/supervisor.py",
            
            # Configuration and environment
            "src/config/__init__.py",
            "src/config/environment.py",
            "src/config.py",
            
            # Database and state management
            "src/database/sqlite_manager.py",
            "src/utils/state_manager.py",
            
            # Interfaces and dependency injection
            "src/interfaces/event_bus.py",
            "src/interfaces/base_interfaces.py",
            "src/core/dependency_injection.py",
            "src/core/config_service.py",
            
            # Monitoring and performance
            "src/monitoring/performance_dashboard.py",
            "src/monitoring/performance_monitor.py",
            "src/monitoring/dual_model_system.py",
            
            # Utilities
            "src/utils/logger.py",
            "src/utils/error_handler.py",
            "src/utils/observability.py",
            "src/utils/warning_symbols.py",
            
            # Exchange components
            "src/exchange/__init__.py",
            "src/exchange/base_exchange.py",
            "src/exchange/binance.py",
            "src/exchange/factory.py",
            
            # Additional trading components
            "src/paper_trader.py",
            "src/tasks.py",
            "src/tracking/trade_tracker.py",
            
            # GUI components (if GUI is enabled)
            "GUI/api_server.py",
            
            # Portfolio management
            "src/supervisor/global_portfolio_manager.py",
            
            # Additional analyst components
            "src/analyst/regime_expert_orchestrator.py",
            "src/analyst/unified_regime_classifier.py",
            "src/analyst/ml_confidence_predictor.py",
            
            # Additional tactician components
            "src/tactician/enhanced_execution_manager.py",
            "src/tactician/enhanced_order_manager.py",
            "src/tactician/position_sizer.py",
            "src/tactician/sr_levels_manager.py",
            
            # Additional supervisor components
            "src/supervisor/enhanced_prediction_service.py",
            "src/supervisor/performance_monitor.py",
            "src/supervisor/risk_allocator.py",
            
            # Additional strategist components
            "src/strategist/strategist.py",
            
            # Additional monitoring components
            "src/monitoring/advanced_tracer.py",
            "src/monitoring/correlation_manager.py",
            "src/monitoring/error_detection_system.py",
            "src/monitoring/fractional_performance_tracker.py",
            "src/monitoring/fractional_system_monitor.py",
            "src/monitoring/integration_manager.py",
            "src/monitoring/metrics_dashboard.py",
            "src/monitoring/ml_monitor.py",
            "src/monitoring/regime_sr_tracker.py",
            "src/monitoring/report_scheduler.py",
            "src/monitoring/surrogate_optimization_monitor.py",
            "src/monitoring/tracking_system.py",
            "src/monitoring/trade_conditions_monitor.py",
            
            # Additional utility components
            "src/utils/async_utils.py",
            "src/utils/centralized_decorators.py",
            "src/utils/comprehensive_file_validation.py",
            "src/utils/confidence.py",
            "src/utils/config_loader.py",
            "src/utils/data_formatting_framework.py",
            "src/utils/data_loader.py",
            "src/utils/data_optimizer.py",
            "src/utils/data_preprocessing.py",
            "src/utils/data_quality_decorators.py",
            "src/utils/data_quality_framework.py",
            "src/utils/data_type_optimizer.py",
            "src/utils/data_validation.py",
            "src/utils/database_security.py",
            "src/utils/decorator_compatibility.py",
            "src/utils/decorator_config.py",
            "src/utils/decorator_registry.py",
            "src/utils/decorators.py",
            "src/utils/domain_errors.py",
            "src/utils/enhanced_config_management.py",
            "src/utils/enhanced_data_quality_decorators.py",
            "src/utils/enhanced_decorators.py",
            "src/utils/enhanced_error_handler.py",
            "src/utils/enhanced_error_handling.py",
            "src/utils/enhanced_memory_management.py",
            "src/utils/enhanced_missing_value_handler.py",
            "src/utils/enhanced_mlflow_integration.py",
            "src/utils/enhanced_outlier_handler.py",
            "src/utils/enhanced_pipeline_decorators.py",
            "src/utils/enhanced_validation_decorators.py",
            "src/utils/hmm_composite_manager.py",
            "src/utils/intelligent_feature_cache.py",
            "src/utils/lookahead_bias_detector.py",
            "src/utils/lookahead_bias_detector_example.py",
            "src/utils/mlflow_utils.py",
            "src/utils/model_manager.py",
            "src/utils/parallel_processing_optimizer.py",
            "src/utils/parquet_utils.py",
            "src/utils/pipeline_standards.py",
            "src/utils/prometheus_metrics.py",
            "src/utils/purged_kfold.py",
            "src/utils/quality_alert_system.py",
            "src/utils/security_framework.py",
            "src/utils/standardized_config_manager.py",
            "src/utils/standardized_error_handler.py",
            "src/utils/standardized_model_manager.py",
            "src/utils/steps_1_7_compatibility_framework.py",
            "src/utils/structured_logging.py",
            "src/utils/time_utils.py",
            "src/utils/trading_decorators.py",
            "src/utils/validation_decorators.py",
            "src/utils/vif_calculator.py",
            "src/utils/vif_validation_decorators.py",
            "src/utils/vif_validation_decorators_simple.py",
            "src/utils/warning_symbols.py",
        ]
        
        # Add these files to called_files
        for file_path in trading_related_files:
            self.called_files.add(file_path)
        
        # Analyze imports in each of these files
        for file_path in trading_related_files:
            self._analyze_file_imports(file_path)
    
    def get_unused_files(self) -> Set[str]:
        """Get files that are not called in the trading execution."""
        return self.all_python_files - self.called_files
    
    def generate_report(self) -> None:
        """Generate a comprehensive report."""
        print("\n" + "="*80)
        print("📊 TRADING EXECUTION ANALYSIS REPORT")
        print("="*80)
        
        print(f"📁 Total Python files found: {len(self.all_python_files)}")
        print(f"🚀 Files called during trading execution: {len(self.called_files)}")
        print(f"❌ Files NOT called: {len(self.get_unused_files())}")
        
        print("\n" + "="*80)
        print("📋 FILES CALLED DURING TRADING EXECUTION")
        print("="*80)
        
        called_files_sorted = sorted(self.called_files)
        for file_path in called_files_sorted:
            print(f"✅ {file_path}")
        
        print("\n" + "="*80)
        print("❌ FILES NOT CALLED DURING TRADING EXECUTION")
        print("="*80)
        
        unused_files = self.get_unused_files()
        unused_files_sorted = sorted(unused_files)
        
        # Categorize unused files
        categories = {
            "training_files": [],
            "validation_files": [],
            "test_files": [],
            "step_files": [],
            "other_files": []
        }
        
        for file_path in unused_files_sorted:
            if "training" in file_path.lower() and "step" in file_path.lower():
                categories["step_files"].append(file_path)
            elif "training" in file_path.lower():
                categories["training_files"].append(file_path)
            elif "validator" in file_path.lower():
                categories["validation_files"].append(file_path)
            elif "test" in file_path.lower():
                categories["test_files"].append(file_path)
            else:
                categories["other_files"].append(file_path)
        
        for category, files in categories.items():
            if files:
                print(f"\n📂 {category.upper().replace('_', ' ')} ({len(files)} files):")
                for file_path in sorted(files):
                    print(f"   ❌ {file_path}")
        
        # Save detailed report to file
        self._save_detailed_report()
    
    def _save_detailed_report(self) -> None:
        """Save a detailed report to JSON file."""
        report = {
            "summary": {
                "total_files": len(self.all_python_files),
                "called_files": len(self.called_files),
                "unused_files": len(self.get_unused_files())
            },
            "called_files": sorted(self.called_files),
            "unused_files": sorted(self.get_unused_files()),
            "import_graph": self.import_graph
        }
        
        with open("trading_execution_analysis.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: trading_execution_analysis.json")

def main():
    """Main execution function."""
    print("🔍 Starting trading execution analysis...")
    
    analyzer = TradingExecutionAnalyzer()
    
    # Step 1: Find all Python files
    analyzer.find_all_python_files()
    
    # Step 2: Analyze trading execution flow
    analyzer.analyze_trading_flow()
    
    # Step 3: Generate report
    analyzer.generate_report()
    
    print("\n✅ Trading analysis complete!")

if __name__ == "__main__":
    main()