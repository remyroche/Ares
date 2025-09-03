#!/usr/bin/env python3
"""
Visualization script for code interactions.
Run this to generate graphical representations.
"""

import json
from pathlib import Path

# Data extracted from the analysis
undefined_functions = {
    {'run_async': 1, 'run_full_training': 1, 'append': 1, 'exception': 1, 'get_exchange': 1, 'lower': 1, 'create_task': 1, 'now': 1, 'total_seconds': 1, 'sleep': 1, 'DataFrame': 1, 'isoformat': 1, 'get_dashboard_summary': 1, 'insert': 1, 'ArgumentParser': 1, 'add_argument': 1, 'parse_args': 1, 'upper': 1, 'copy': 1, 'items': 1}

}

missing_await_functions = {
    {'run_training': 1, 'main': 22, 'setup_paper_trader': 1, 'setup_performance_reporter': 1, 'setup_enhanced_training_manager': 1, 'run_integration_example': 1, 'demonstrate_gpu_integration': 1, 'create_sample_data': 1, 'load_config': 4, 'step1_precompute_features': 1, 'step2_run_backtests': 1, 'step3_performance_comparison': 1, 'step4_cache_management': 1, 'run_step': 19, 'test': 17, 'create_sr_levels_manager': 1, 'run_validator': 22, 'test_validator': 12, '_main': 1, 'run_validation': 1, 'run_step_enhanced': 2, '_test': 2, '_create_basic_features': 1, 'setup_sr_detection_optimizer': 1, 'load_unified_data': 3, 'run_integrated_pipeline': 1, '_save_optimization_results': 2, '_generate_optimization_report': 2, 'download_all_data_with_consolidation': 3, '_load_regime_data': 2, 'setup_sr_optuna_optimizer': 1, 'setup_regime_specific_optimizer': 1, 'start_data_quality_dashboard': 1, 'run_comprehensive_gap_filling_pipeline': 1, 'run_gap_filling_pipeline': 1, 'stop': 6, '_execute_pipeline_function': 2, 'get_step_reports': 1, '_optimize_data_types': 1, '_remove_unnecessary_columns': 1, '_optimize_index': 1, '_optimize_memory_usage': 1, 'optimize_dataframe': 1, 'initialize': 35, 'setup_dual_model_system': 1, 'validate_migration_file': 2, 'export_database_for_trading': 1, 'import_database_for_trading': 1, '_attempt_recovery': 9}

}

print("CODE INTERACTION VISUALIZATION DATA")
print("=" * 50)
print()
print("Top 10 Undefined Functions:")
for func, count in sorted(undefined_functions.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {func}: {count} occurrences")

print()
print("Top 10 Async Functions Missing Await:")
for func, count in sorted(missing_await_functions.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {func}: {count} occurrences")

print()
print("To create visual graphs:")
print("1. Use matplotlib/seaborn for bar charts of function counts")
print("2. Use networkx for dependency graphs")
print("3. Use graphviz for call flow diagrams")
print()
print("Example visualization code:")
print("""
import matplotlib.pyplot as plt

# Bar chart of undefined functions
funcs = list(undefined_functions.keys())[:10]
counts = [undefined_functions[f] for f in funcs]

plt.figure(figsize=(12, 6))
plt.bar(funcs, counts)
plt.xticks(rotation=45, ha='right')
plt.title('Top 10 Undefined Functions')
plt.xlabel('Function Name')
plt.ylabel('Occurrences')
plt.tight_layout()
plt.savefig('undefined_functions.png')
""")
