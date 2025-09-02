#!/usr/bin/env python3
"""
Comprehensive Import Analyzer
Analyzes all imports across the codebase with detailed reporting.
"""

import ast
import os
import sys
import re
from pathlib import Path
from collections import defaultdict, Counter
import json
from typing import Dict, Set, List, Tuple, Any

class ComprehensiveImportAnalyzer:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.all_imports = defaultdict(set)  # file -> imports
        self.all_from_imports = defaultdict(set)  # file -> from imports
        self.import_frequency = Counter()  # import -> count
        self.from_import_frequency = Counter()  # from import -> count
        self.import_locations = defaultdict(list)  # import -> [files]
        self.unused_imports = defaultdict(set)  # file -> unused imports
        self.syntax_errors = defaultdict(list)
        self.external_modules = set()
        self.standard_lib_modules = set()
        self.internal_modules = set()
        self.third_party_modules = set()
        
        # Standard library modules (Python 3.7+)
        self.standard_lib = {
            'abc', 'argparse', 'asyncio', 'collections', 'contextlib', 'copy',
            'datetime', 'decimal', 'enum', 'functools', 'glob', 'gzip',
            'hashlib', 'importlib', 'inspect', 'io', 'itertools', 'json',
            'logging', 'math', 'multiprocessing', 'operator', 'os', 'pathlib',
            'pickle', 'random', 're', 'shutil', 'signal', 'socket', 'sqlite3',
            'statistics', 'string', 'subprocess', 'sys', 'tempfile', 'threading',
            'time', 'traceback', 'typing', 'unittest', 'urllib', 'uuid',
            'warnings', 'weakref', 'zipfile', 'zlib'
        }
        
    def find_python_files(self):
        """Find all Python files in the repository."""
        python_files = []
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env']]
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        return python_files
    
    def is_standard_lib(self, module_name):
        """Check if a module is part of Python standard library."""
        return module_name in self.standard_lib
    
    def categorize_module(self, module_name):
        """Categorize a module as standard lib, third party, or internal."""
        if self.is_standard_lib(module_name):
            self.standard_lib_modules.add(module_name)
            return "standard_lib"
        elif module_name.startswith('src.') or module_name in ['src']:
            self.internal_modules.add(module_name)
            return "internal"
        else:
            self.third_party_modules.add(module_name)
            return "third_party"
    
    def extract_imports_with_ast(self, content):
        """Extract imports using AST parsing."""
        imports = set()
        from_imports = set()
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        imports.add(module_name)
                        if alias.asname:
                            imports.add(alias.asname)
                            
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        imports.add(module_name)
                        for alias in node.names:
                            if alias.asname:
                                imports.add(alias.asname)
                            else:
                                from_imports.add(alias.name)
                                
        except SyntaxError as e:
            raise e
        
        return imports, from_imports
    
    def extract_imports_with_regex(self, content):
        """Extract imports using regex when AST fails."""
        imports = set()
        from_imports = set()
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Handle import statements
            if line.startswith('import '):
                # Match: import module, import module as alias, import module.submodule
                match = re.match(r'^import\s+([a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*)(?:\s+as\s+([a-zA-Z_][a-zA-Z0-9_]*))?', line)
                if match:
                    module_name = match.group(1).split('.')[0]
                    imports.add(module_name)
                    if match.group(3):  # alias
                        imports.add(match.group(3))
            
            # Handle from imports
            elif line.startswith('from '):
                # Match: from module import item, from module import item as alias
                match = re.match(r'^from\s+([a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import\s+(.+)', line)
                if match:
                    module_name = match.group(1).split('.')[0]
                    imports.add(module_name)
                    
                    # Parse the imported items
                    items_part = match.group(3)
                    if items_part:
                        # Handle: import item1, item2, item3
                        items = [item.strip() for item in items_part.split(',')]
                        for item in items:
                            # Handle: item as alias
                            if ' as ' in item:
                                item_name, alias = item.split(' as ')
                                from_imports.add(item_name.strip())
                                from_imports.add(alias.strip())
                            else:
                                from_imports.add(item.strip())
        
        return imports, from_imports
    
    def analyze_file_imports(self, file_path):
        """Analyze imports in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try AST parsing first
            try:
                imports, from_imports = self.extract_imports_with_ast(content)
            except SyntaxError:
                # Fallback to regex parsing
                imports, from_imports = self.extract_imports_with_regex(content)
                self.syntax_errors[file_path].append("Used regex fallback due to syntax errors")
            
            # Store imports
            if imports:
                self.all_imports[file_path] = imports
                for imp in imports:
                    self.import_frequency[imp] += 1
                    self.import_locations[imp].append(str(file_path))
                    self.categorize_module(imp)
            
            if from_imports:
                self.all_from_imports[file_path] = from_imports
                for imp in from_imports:
                    self.from_import_frequency[imp] += 1
            
            return True
            
        except Exception as e:
            self.syntax_errors[file_path].append(f"Error reading file: {e}")
            return False
    
    def find_unused_imports(self):
        """Find imports that are never used in the file."""
        for file_path in self.find_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Get all imported items
                all_imported = set()
                if file_path in self.all_imports:
                    all_imported.update(self.all_imports[file_path])
                if file_path in self.all_from_imports:
                    all_imported.update(self.all_from_imports[file_path])
                
                if not all_imported:
                    continue
                
                # Find all used names in the file
                used_names = set()
                
                # Simple regex-based usage detection
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Look for variable/function usage
                        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', line)
                        used_names.update(words)
                
                # Find unused imports
                unused = all_imported - used_names
                if unused:
                    self.unused_imports[file_path] = unused
                    
            except Exception:
                continue
    
    def analyze_all_files(self):
        """Analyze all Python files for imports."""
        python_files = self.find_python_files()
        
        print(f"🔍 Analyzing imports in {len(python_files)} Python files...")
        
        for i, file_path in enumerate(python_files):
            if i % 50 == 0:
                print(f"Processing file {i+1}/{len(python_files)}...")
            
            self.analyze_file_imports(file_path)
        
        # Find unused imports
        print("Finding unused imports...")
        self.find_unused_imports()
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive import report."""
        report = {
            "summary": {
                "total_files_analyzed": len(self.all_imports),
                "total_import_statements": sum(len(imports) for imports in self.all_imports.values()),
                "total_from_imports": sum(len(imports) for imports in self.all_from_imports.values()),
                "unique_modules_imported": len(self.import_frequency),
                "unique_items_from_imported": len(self.from_import_frequency),
                "files_with_syntax_errors": len(self.syntax_errors),
                "files_with_unused_imports": len(self.unused_imports)
            },
            "import_categories": {
                "standard_library": list(sorted(self.standard_lib_modules)),
                "third_party": list(sorted(self.third_party_modules)),
                "internal": list(sorted(self.internal_modules))
            },
            "most_imported_modules": [
                {"module": module, "count": count, "files": self.import_locations[module]}
                for module, count in self.import_frequency.most_common(50)
            ],
            "most_from_imported_items": [
                {"item": item, "count": count}
                for item, count in self.from_import_frequency.most_common(50)
            ],
            "import_locations": {
                module: files for module, files in self.import_locations.items()
            },
            "unused_imports": {
                str(k): list(v) for k, v in self.unused_imports.items()
            },
            "syntax_errors": {
                str(k): v for k, v in self.syntax_errors.items()
            },
            "files_with_most_imports": [
                {
                    "file": str(file_path),
                    "import_count": len(imports),
                    "from_import_count": len(self.all_from_imports.get(file_path, set())),
                    "total_imports": len(imports) + len(self.all_from_imports.get(file_path, set())),
                    "imports": list(imports),
                    "from_imports": list(self.all_from_imports.get(file_path, set()))
                }
                for file_path, imports in sorted(
                    self.all_imports.items(),
                    key=lambda x: len(x[1]) + len(self.all_from_imports.get(x[0], set())),
                    reverse=True
                )
            ]
        }
        
        return report
    
    def print_comprehensive_report(self):
        """Print a comprehensive import report."""
        report = self.generate_comprehensive_report()
        summary = report["summary"]
        
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE IMPORT ANALYSIS REPORT")
        print(f"{'='*80}")
        print(f"📁 Total files analyzed: {summary['total_files_analyzed']}")
        print(f"📦 Total import statements: {summary['total_import_statements']}")
        print(f"📥 Total from-imports: {summary['total_from_imports']}")
        print(f"🔗 Unique modules imported: {summary['unique_modules_imported']}")
        print(f"📋 Unique items from-imported: {summary['unique_items_from_imported']}")
        print(f"⚠️  Files with syntax errors: {summary['files_with_syntax_errors']}")
        print(f"🗑️  Files with unused imports: {summary['files_with_unused_imports']}")
        
        print(f"\n📊 IMPORT CATEGORIES:")
        print(f"   • Standard library: {len(report['import_categories']['standard_library'])} modules")
        print(f"   • Third party: {len(report['import_categories']['third_party'])} modules")
        print(f"   • Internal: {len(report['import_categories']['internal'])} modules")
        
        print(f"\n🔝 TOP 20 MOST IMPORTED MODULES:")
        for i, module_info in enumerate(report['most_imported_modules'][:20], 1):
            print(f"   {i:2d}. {module_info['module']}: imported {module_info['count']} times in {len(module_info['files'])} files")
        
        print(f"\n📥 TOP 20 MOST FROM-IMPORTED ITEMS:")
        for i, item_info in enumerate(report['most_from_imported_items'][:20], 1):
            print(f"   {i:2d}. {item_info['item']}: imported {item_info['count']} times")
        
        print(f"\n📁 FILES WITH MOST IMPORTS:")
        for i, file_info in enumerate(report['files_with_most_imports'][:15], 1):
            print(f"   {i:2d}. {file_info['file']}: {file_info['total_imports']} total imports")
            print(f"       • Import statements: {file_info['import_count']}")
            print(f"       • From-imports: {file_info['from_import_count']}")
        
        if len(report['files_with_most_imports']) > 15:
            print(f"   ... and {len(report['files_with_most_imports']) - 15} more files")
        
        print(f"\n🗑️  UNUSED IMPORTS SUMMARY:")
        total_unused = sum(len(unused) for unused in self.unused_imports.values())
        print(f"   • Total unused imports: {total_unused}")
        print(f"   • Files with unused imports: {len(self.unused_imports)}")
        
        if self.unused_imports:
            print(f"\n📋 TOP 10 FILES WITH MOST UNUSED IMPORTS:")
            sorted_unused = sorted(
                self.unused_imports.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )
            for i, (file_path, unused) in enumerate(sorted_unused[:10], 1):
                print(f"   {i:2d}. {file_path}: {len(unused)} unused imports")
                print(f"       • {', '.join(list(unused)[:5])}")
                if len(unused) > 5:
                    print(f"       • ... and {len(unused) - 5} more")
        
        print(f"\n💡 IMPORT ANALYSIS INSIGHTS:")
        print(f"   • Most common import: {report['most_imported_modules'][0]['module'] if report['most_imported_modules'] else 'None'}")
        print(f"   • Most complex file: {report['files_with_most_imports'][0]['file'] if report['files_with_most_imports'] else 'None'}")
        print(f"   • Import diversity: {len(self.import_frequency)} unique modules")
        print(f"   • Cleanup potential: {total_unused} unused imports can be removed")
    
    def save_comprehensive_report(self, output_path="comprehensive_import_report.json"):
        """Save comprehensive import report as JSON."""
        report = self.generate_comprehensive_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Comprehensive import report saved to: {output_path}")
    
    def generate_import_heatmap_data(self):
        """Generate data for import heatmap visualization."""
        # Create a matrix of file vs import relationships
        all_files = list(self.all_imports.keys())
        all_modules = list(self.import_frequency.keys())
        
        # Create CSV for heatmap
        csv_content = ["File,Module,Imported"]
        
        for file_path in all_files:
            file_str = str(file_path)
            for module in all_modules:
                if module in self.all_imports[file_path]:
                    csv_content.append(f"{file_str},{module},1")
                else:
                    csv_content.append(f"{file_str},{module},0")
        
        with open("import_heatmap.csv", 'w') as f:
            f.write('\n'.join(csv_content))
        
        print("✅ Import heatmap data saved to import_heatmap.csv")

def main():
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "src"
    
    print(f"🔍 Comprehensive import analysis for: {root_dir}")
    
    analyzer = ComprehensiveImportAnalyzer(root_dir)
    analyzer.analyze_all_files()
    
    # Generate and print comprehensive report
    analyzer.print_comprehensive_report()
    
    # Save detailed report
    analyzer.save_comprehensive_report()
    
    # Generate heatmap data
    analyzer.generate_import_heatmap_data()
    
    print(f"\n{'='*80}")
    print(f"IMPORT ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"📊 Generated comprehensive import report")
    print(f"📈 Created import heatmap data")
    print(f"💡 Review the report for import optimization opportunities")

if __name__ == "__main__":
    main()