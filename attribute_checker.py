#!/usr/bin/env python3
"""
Enhanced attribute access checker to detect missing methods/attributes with reduced false positives
"""

import ast
import sys
from pathlib import Path
from typing import Set, Dict, List

class AttributeChecker(ast.NodeVisitor):
    def __init__(self, class_name: str):
        self.class_name = class_name
        self.class_methods: Set[str] = set()
        self.instance_attrs: Set[str] = set()  # Attributes set in __init__
        self.attribute_accesses: List[Dict] = []
        self.current_class = None
        self.all_classes_in_file: Set[str] = set()  # Track all classes in the file
        self.class_hierarchy: Dict[str, str] = {}  # Track inheritance

        # Common attributes that are typically injected or inherited
        self.known_external_attrs = {
            # Logging
            'logger',
            # Configuration
            'config', 'sr_optimization_config',
            # Services/Dependencies
            'dependency_container', 'parquet_utils', 'json_serializer',
            'data_validator', 'data_cleaner', 'data_transformer',
            'm1_gpu_manager', 'm1_memory_optimizer', 'm1_cpu_optimizer',
            'universal_serializer',
            # Base class attributes
            'standards', 'debug_mode', 'enable_fast_fail', 'fast_fail_on_ml_errors',
            'max_ml_failures', 'enable_hyperparameter_optimization', 'optimization_method',
            'optimization_folds', 'optimization_trials', 'enable_walk_forward_validation',
            'walk_forward_folds', 'walk_forward_test_size', 'enable_m1_optimizations',
            'enable_memory_optimization', 'enable_parallel_processing',
            # Performance monitoring
            'start_time', 'performance_monitor',
            # Error handling
            'ml_failure_count', 'ml_failure_reasons',
            # Results storage
            '_hyperparameter_results', '_walk_forward_results',
            # Other common attributes
            'name', 'check_memory_func', 'start_memory',
            '_log_buffer', '_run_sr_calculation',
            # Common function attributes
            'features_data', 'ml_model_configs',
        }

        # Methods that might be defined in parent classes or mixins
        self.known_parent_methods = {
            'initialize_services', 'execute_logic', 'execute_main_logic',
            '_get_fallback_sr_levels', '_get_fallback_ml_result', '_get_default_hyperparameters',
            '_validate_sr_results', '_prepare_sr_targets', '_prepare_ml_features',
            '_optimize_hyperparameters', '_walk_forward_validation', '_optimize_feature_selection',
            '_train_multiple_models', '_optimize_hyperparameters_async', '_perform_cross_validation',
            '_calculate_evaluation_metrics', '_save_best_model', '_chunked_hyperparameter_optimization',
            '_m1_gpu_hyperparameter_optimization', '_m1_cpu_hyperparameter_optimization',
            '_simplified_hyperparameter_optimization', '_optimized_grid_search_optimization',
            '_optimized_random_search_optimization', '_optimized_bayesian_optimization',
            '_halving_search_optimization', '_combine_optimization_results',
            '_create_optimization_tasks', '_create_model_training_tasks', '_combine_model_results',
            '_get_fallback_feature_selection_info', '_aggressive_feature_reduction',
            '_fast_feature_filtering', '_incremental_rf_feature_selection',
            '_compute_mutual_information_scores', '_compute_shap_importance_optimized',
            '_apply_numpy_compatibility_patch', '_compute_permutation_importance',
            '_calculate_advanced_momentum_features', '_validate_and_fill_features',
            '_calculate_correlation_features', '_calculate_liquidity_features',
            '_calculate_adaptive_features', '_calculate_price_distance', '_assess_risk',
            '_validate_and_fix_input_data', '_engineer_features', '_get_error_recommendations',
            '_get_troubleshooting_steps', '_get_next_actions', '_flush_log_buffer',
            '_generate_final_report', '_format_sr_levels_for_pipeline', '_train_ml_models_chunked_optimized',
            '_consolidate_sr_levels', '_merge_level_group', '_run_sr_detection',
            '_aggregate_chunk_results', '_train_ml_models', '_handle_ml_failure',
            '_get_fallback_ml_result_with_failure_info', '_log_utility_integration_status',
            '_initialize_step', '_initialize_logging_verbosity', '_robust_error_handling',
            '_reduce_data_size', '_get_fallback_result', '_performance_monitor',
            '_optimized_logging', '_reduce_logging_verbosity', '_clear_cache',
            '_get_cache_stats', '_run_sr_detection_with_fast_fail',
        }

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Track all classes in the file
        self.all_classes_in_file.add(node.name)

        # Track inheritance
        if node.bases:
            for base in node.bases:
                if isinstance(base, ast.Name):
                    self.class_hierarchy[node.name] = base.id

        if node.name == self.class_name:
            self.current_class = node.name
            # Visit class body to collect methods and instance attributes
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    self.class_methods.add(item.name)
                elif isinstance(item, ast.AsyncFunctionDef):
                    self.class_methods.add(item.name)
                # Look for instance attribute assignments in __init__
                if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                    self._collect_instance_attrs(item)
            self.generic_visit(node)
            self.current_class = None
        else:
            self.generic_visit(node)

    def _collect_instance_attrs(self, init_node: ast.FunctionDef) -> None:
        """Collect instance attributes set in __init__ method."""
        for node in ast.walk(init_node):
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == 'self':
                    if isinstance(node.ctx, ast.Store):
                        self.instance_attrs.add(node.attr)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == 'self':
            # This is a self.attribute access
            attr_name = node.attr
            # Skip known external attributes and instance attributes
            if attr_name not in self.known_external_attrs and attr_name not in self.instance_attrs:
                self.attribute_accesses.append({
                    'line': node.lineno,
                    'column': node.col_offset,
                    'attribute': attr_name,
                    'type': 'attribute_access'
                })
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'self':
                # This is a self.method() call
                method_name = node.func.attr

                # Skip if it's a known external method or already in our class
                if method_name in self.known_parent_methods or method_name in self.class_methods:
                    pass  # This is fine
                elif method_name.startswith('_') and len(method_name) > 1:
                    # Private methods - these might be real issues but could also be defined elsewhere
                    # Only flag if they seem like they should be in this class
                    if not any(method_name in known for known in [self.known_parent_methods, self.class_methods]):
                        self.attribute_accesses.append({
                            'line': node.lineno,
                            'column': node.func.col_offset,
                            'attribute': method_name,
                            'type': 'method_call',
                            'severity': 'warning'
                        })
                else:
                    # Public methods - these are more likely to be real issues
                    self.attribute_accesses.append({
                        'line': node.lineno,
                        'column': node.func.col_offset,
                        'attribute': method_name,
                        'type': 'method_call',
                        'severity': 'error'
                    })
        self.generic_visit(node)

def check_file(file_path: str, class_name: str) -> Dict:
    """Check a file for attribute access issues."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content, filename=file_path)
        checker = AttributeChecker(class_name)
        checker.visit(tree)

        # Find missing methods/attributes with severity filtering
        missing_items = []
        warnings = []
        errors = []

        for access in checker.attribute_accesses:
            if access['attribute'] not in checker.class_methods:
                # Check if it's a common attribute that might be defined elsewhere
                if access['attribute'] not in ['__init__', '__str__', '__repr__', '__eq__', '__hash__']:
                    missing_items.append(access)
                    severity = access.get('severity', 'warning')
                    if severity == 'error':
                        errors.append(access)
                    else:
                        warnings.append(access)

        return {
            'status': 'success',
            'class_methods': list(checker.class_methods),
            'attribute_accesses': checker.attribute_accesses,
            'missing_items': missing_items,
            'warnings': warnings,
            'errors': errors,
            'total_accesses': len(checker.attribute_accesses),
            'missing_count': len(missing_items),
            'warning_count': len(warnings),
            'error_count': len(errors),
            'all_classes_in_file': list(checker.all_classes_in_file),
            'class_hierarchy': checker.class_hierarchy
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python attribute_checker.py <file_path> <class_name>")
        sys.exit(1)

    file_path = sys.argv[1]
    class_name = sys.argv[2]

    result = check_file(file_path, class_name)

    if result['status'] == 'error':
        print(f"Error: {result['error']}")
        sys.exit(1)

    print(f"\nAttribute Access Analysis for class '{class_name}':")
    print(f"Total methods found: {len(result['class_methods'])}")
    print(f"Total attribute accesses: {result['total_accesses']}")
    print(f"Potentially missing items: {result['missing_count']}")
    print(f"  - Warnings: {result['warning_count']}")
    print(f"  - Errors: {result['error_count']}")

    print(f"\nClasses in file: {', '.join(result['all_classes_in_file'])}")
    if result['class_hierarchy']:
        print(f"Class hierarchy: {result['class_hierarchy']}")

    if result['errors']:
        print("\n🚨 CRITICAL ISSUES (Errors):")
        for item in result['errors']:
            print(f"  Line {item['line']}: {item['type']} - '{item['attribute']}'")

    if result['warnings']:
        print("\n⚠️  POTENTIAL ISSUES (Warnings):")
        for item in result['warnings']:
            print(f"  Line {item['line']}: {item['type']} - '{item['attribute']}'")

    if result['missing_count'] == 0:
        print("\n✅ No obvious missing methods/attributes found")
    else:
        print(f"\n📊 Summary: {result['error_count']} errors, {result['warning_count']} warnings")
        if result['error_count'] > 0:
            print("❌ Action required: Fix critical errors first")
        else:
            print("ℹ️  These may be false positives - review manually")
