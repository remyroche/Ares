#!/usr/bin/env python3
"""Import analyzer for import relationship analysis."""

import ast
from typing import Dict, List, Any
from .base_analyzer import BaseAnalyzer
import numpy as np


class ImportAnalyzer(BaseAnalyzer):
    """Analyzes import relationships."""
    
    def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """Analyze imports in the directory."""
        python_files = self._find_python_files(directory_path)
        files = {}
        circular_imports = []
        
        for file_path in python_files:
            content = self._read_file_safely(file_path)
            if not content:
                continue
            
            tree = self._parse_ast_safely(content, file_path)
            if not tree:
                continue
            
            imports = self._extract_imports(tree)
            files[str(file_path)] = {
                "imports": imports,
                "import_count": len(imports)
            }
            
            self.stats["files_analyzed"] += 1
            self.stats["total_items"] += len(imports)
        
        return {
            "files": files,
            "circular_imports": circular_imports,
            "total_imports": sum(len(f["imports"]) for f in files.values()),
            "stats": self.stats
        }
    
    def _extract_imports(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract import statements from AST."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({"module": alias.name, "type": "import"})
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append({"module": node.module, "type": "from_import"})
        return imports
