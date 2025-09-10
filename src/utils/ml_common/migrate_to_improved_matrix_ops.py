#!/usr/bin/env python3
"""
Migration script for transitioning to improved matrix operations.

This script helps users migrate from the old matrix operations to the new
M1-optimized version with minimal code changes.
"""

import ast
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

class MatrixOperationsMigrator:
    """Migrates code to use improved matrix operations."""
    
    def __init__(self):
        self.old_imports = [
            'from src.utils.enhanced_matrix_operations import',
            'from src.utils.ml_common.matrix_operations import get_enhanced_matrix_operations'
        ]
        
        self.new_imports = [
            'from src.utils.ml_common.matrix_operations import get_enhanced_matrix_operations',
            'from src.utils.ml_common.matrix_operations import m1_matrix_multiply, m1_batch_process, m1_correlation_matrix'
        ]
        
        self.function_mappings = {
            'gpu_matrix_multiply': 'm1_matrix_multiply',
            'correlation_matrix_gpu': 'm1_correlation_matrix',
            'eigendecomposition_gpu': 'm1_eigendecomposition',
            'svd_gpu': 'm1_svd_decomposition'
        }
        
        self.pattern_replacements = [
            # Replace old function calls with new ones
            (r'gpu_matrix_multiply\(', 'm1_matrix_multiply('),
            (r'correlation_matrix_gpu\(', 'm1_correlation_matrix('),
            (r'eigendecomposition_gpu\(', 'm1_eigendecomposition('),
            (r'svd_gpu\(', 'm1_svd_decomposition('),
            
            # Replace old imports
            (r'from src\.utils\.enhanced_matrix_operations import', 
             'from src.utils.ml_common.matrix_operations import'),
        ]

    def find_python_files(self, directory: str) -> List[Path]:
        """Find all Python files in a directory."""
        python_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        return python_files

    def analyze_file(self, file_path: Path) -> Dict[str, any]:
        """Analyze a Python file for matrix operations usage."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            analysis = {
                'file_path': file_path,
                'has_old_imports': False,
                'has_matrix_operations': False,
                'old_functions_used': [],
                'suggested_changes': [],
                'lines_to_change': []
            }
            
            # Check for old imports
            for line_num, line in enumerate(content.split('\n'), 1):
                for old_import in self.old_imports:
                    if old_import in line:
                        analysis['has_old_imports'] = True
                        analysis['lines_to_change'].append({
                            'line_num': line_num,
                            'line': line.strip(),
                            'type': 'import',
                            'suggestion': self._suggest_import_replacement(line)
                        })
            
            # Check for old function usage
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name in self.function_mappings:
                            analysis['has_matrix_operations'] = True
                            analysis['old_functions_used'].append(func_name)
                            analysis['suggested_changes'].append({
                                'old_function': func_name,
                                'new_function': self.function_mappings[func_name],
                                'suggestion': f"Replace {func_name} with {self.function_mappings[func_name]}"
                            })
            
            return analysis
            
        except Exception as e:
            return {
                'file_path': file_path,
                'error': str(e),
                'has_old_imports': False,
                'has_matrix_operations': False,
                'old_functions_used': [],
                'suggested_changes': [],
                'lines_to_change': []
            }

    def _suggest_import_replacement(self, line: str) -> str:
        """Suggest replacement for old import."""
        if 'enhanced_matrix_operations' in line:
            return line.replace('enhanced_matrix_operations', 'matrix_operations')
        return line

    def generate_migration_report(self, directory: str) -> str:
        """Generate a migration report for a directory."""
        python_files = self.find_python_files(directory)
        
        report = []
        report.append("# Matrix Operations Migration Report")
        report.append("=" * 50)
        report.append(f"Analyzed {len(python_files)} Python files in {directory}")
        report.append("")
        
        files_needing_migration = []
        total_old_functions = 0
        
        for file_path in python_files:
            analysis = self.analyze_file(file_path)
            
            if analysis.get('has_old_imports') or analysis.get('has_matrix_operations'):
                files_needing_migration.append(analysis)
                total_old_functions += len(analysis.get('old_functions_used', []))
                
                report.append(f"## {file_path}")
                report.append("")
                
                if analysis.get('error'):
                    report.append(f"❌ Error analyzing file: {analysis['error']}")
                    report.append("")
                    continue
                
                if analysis.get('has_old_imports'):
                    report.append("🔍 **Old imports found:**")
                    for change in analysis.get('lines_to_change', []):
                        if change['type'] == 'import':
                            report.append(f"  - Line {change['line_num']}: `{change['line']}`")
                            report.append(f"    → Suggested: `{change['suggestion']}`")
                    report.append("")
                
                if analysis.get('has_matrix_operations'):
                    report.append("🔧 **Old functions found:**")
                    for change in analysis.get('suggested_changes', []):
                        report.append(f"  - `{change['old_function']}` → `{change['new_function']}`")
                    report.append("")
                
                # Add migration suggestions
                report.append("💡 **Migration suggestions:**")
                if analysis.get('has_old_imports'):
                    report.append("  1. Update import statements")
                if analysis.get('has_matrix_operations'):
                    report.append("  2. Replace old function calls with M1-optimized versions")
                report.append("  3. Test performance improvements")
                report.append("")
        
        # Summary
        report.append("## Summary")
        report.append("=" * 20)
        report.append(f"Files needing migration: {len(files_needing_migration)}")
        report.append(f"Total old function calls: {total_old_functions}")
        report.append("")
        
        if files_needing_migration:
            report.append("## Migration Steps")
            report.append("")
            report.append("1. **Update imports:** Replace old import statements")
            report.append("2. **Replace functions:** Use M1-optimized function names")
            report.append("3. **Test performance:** Verify improvements")
            report.append("4. **Update documentation:** Reflect new API usage")
            report.append("")
            
            report.append("## Example Migration")
            report.append("")
            report.append("### Before:")
            report.append("```python")
            report.append("from src.utils.enhanced_matrix_operations import gpu_matrix_multiply")
            report.append("result = gpu_matrix_multiply(matrix_a, matrix_b)")
            report.append("```")
            report.append("")
            report.append("### After:")
            report.append("```python")
            report.append("from src.utils.ml_common.matrix_operations import m1_matrix_multiply")
            report.append("result = m1_matrix_multiply(matrix_a, matrix_b)")
            report.append("```")
        else:
            report.append("✅ No migration needed - all files are up to date!")
        
        return "\n".join(report)

    def create_migration_script(self, directory: str, output_file: str = "migrate_matrix_ops.py"):
        """Create an automated migration script."""
        python_files = self.find_python_files(directory)
        files_to_migrate = []
        
        for file_path in python_files:
            analysis = self.analyze_file(file_path)
            if analysis.get('has_old_imports') or analysis.get('has_matrix_operations'):
                files_to_migrate.append(analysis)
        
        script_content = f'''#!/usr/bin/env python3
"""
Automated migration script for matrix operations.
Generated for directory: {directory}
"""

import os
import re
from pathlib import Path

def migrate_file(file_path: str):
    """Migrate a single file."""
    print(f"Migrating {{file_path}}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply replacements
        replacements = [
            (r'from src\.utils\.enhanced_matrix_operations import', 
             'from src.utils.ml_common.matrix_operations import'),
            (r'gpu_matrix_multiply\(', 'm1_matrix_multiply('),
            (r'correlation_matrix_gpu\(', 'm1_correlation_matrix('),
            (r'eigendecomposition_gpu\(', 'm1_eigendecomposition('),
            (r'svd_gpu\(', 'm1_svd_decomposition('),
        ]
        
        for old_pattern, new_pattern in replacements:
            content = re.sub(old_pattern, new_pattern, content)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ Updated {{file_path}}")
        else:
            print(f"  ⏭️  No changes needed for {{file_path}}")
            
    except Exception as e:
        print(f"  ❌ Error migrating {{file_path}}: {{e}}")

def main():
    """Run migration for all identified files."""
    files_to_migrate = [
'''
        
        for analysis in files_to_migrate:
            script_content += f'        "{analysis["file_path"]}",\n'
        
        script_content += '''    ]
    
    print(f"Migrating {len(files_to_migrate)} files...")
    
    for file_path in files_to_migrate:
        migrate_file(file_path)
    
    print("Migration completed!")
    print("")
    print("Next steps:")
    print("1. Test your code to ensure it works correctly")
    print("2. Check performance improvements")
    print("3. Update any documentation or comments")

if __name__ == "__main__":
    main()
'''
        
        with open(output_file, 'w') as f:
            f.write(script_content)
        
        print(f"Migration script created: {output_file}")
        print(f"Run with: python {output_file}")

def main():
    """Main migration tool."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Matrix Operations Migration Tool")
    parser.add_argument("directory", help="Directory to analyze")
    parser.add_argument("--report", help="Output report file", default="migration_report.md")
    parser.add_argument("--script", help="Create migration script", action="store_true")
    parser.add_argument("--script-file", help="Migration script filename", default="migrate_matrix_ops.py")
    
    args = parser.parse_args()
    
    migrator = MatrixOperationsMigrator()
    
    # Generate report
    report = migrator.generate_migration_report(args.directory)
    with open(args.report, 'w') as f:
        f.write(report)
    print(f"Migration report created: {args.report}")
    
    # Create migration script if requested
    if args.script:
        migrator.create_migration_script(args.directory, args.script_file)

if __name__ == "__main__":
    main()