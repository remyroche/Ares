"""
Code Duplication Analyzer - Detects duplicate code patterns and suggests refactoring opportunities.
"""

import os
import ast
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import defaultdict, Counter
import json
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from minimal_config import CodeQualityConfig, get_default_config
from minimal_file_utils import find_python_files


class DuplicationBlock:
    """Container for duplicate code block information."""
    
    def __init__(self, content: str, hash_value: str, locations: List[Tuple[str, int, int]], 
                 similarity: float = 1.0, block_type: str = "unknown"):
        self.content = content
        self.hash_value = hash_value
        self.locations = locations  # (file_path, start_line, end_line)
        self.similarity = similarity
        self.block_type = block_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "hash_value": self.hash_value,
            "locations": [
                {"file": loc[0], "start_line": loc[1], "end_line": loc[2]}
                for loc in self.locations
            ],
            "similarity": self.similarity,
            "block_type": self.block_type,
            "duplicate_count": len(self.locations)
        }


class CodeDuplicationAnalyzer:
    """
    Comprehensive code duplication detection and analysis.
    
    Features:
    - Exact duplicate detection
    - Similar code pattern detection
    - Function-level duplication analysis
    - Class-level duplication analysis
    - Refactoring suggestions
    - Duplication metrics and reporting
    """
    
    def __init__(self, config: Optional[CodeQualityConfig] = None):
        self.config = config or get_default_config()
        self.duplication_blocks: List[DuplicationBlock] = []
        self.file_duplications: Dict[str, List[DuplicationBlock]] = {}
        self.duplication_metrics: Dict[str, Dict[str, Any]] = {}
        self.refactoring_suggestions: List[str] = []
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a single Python file for code duplication.
        
        Args:
            file_path: Path to Python file to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            file_path = Path(file_path).resolve()
            
            # Clear previous results for this file
            self.duplication_blocks = [block for block in self.duplication_blocks 
                                     if not any(loc[0] == str(file_path) for loc in block.locations)]
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Perform duplication analysis
            function_duplications = self._analyze_function_duplications(tree, str(file_path))
            class_duplications = self._analyze_class_duplications(tree, str(file_path))
            block_duplications = self._analyze_code_blocks(tree, str(file_path))
            pattern_duplications = self._analyze_code_patterns(tree, str(file_path))
            
            # Combine results
            combined_results = {
                "function_duplications": function_duplications,
                "class_duplications": class_duplications,
                "block_duplications": block_duplications,
                "pattern_duplications": pattern_duplications
            }
            
            # Calculate duplication metrics
            duplication_score = self._calculate_duplication_score(combined_results)
            
            # Store results
            self.file_duplications[str(file_path)] = self.duplication_blocks
            
            return {
                "status": "success",
                "issues_found": len(self.duplication_blocks),
                "issues_fixed": 0,
                "details": combined_results,
                "duplication_score": duplication_score
            }
            
        except Exception as e:
            logging.error(f"Error in duplication analysis for {file_path}: {e}")
            return {
                "status": "error",
                "issues_found": 0,
                "issues_fixed": 0,
                "details": {"error": str(e)},
                "duplication_score": 0
            }
    
    def _analyze_function_duplications(self, tree: ast.AST, file_path: str) -> Dict[str, Any]:
        """Analyze function-level duplications."""
        functions = []
        function_hashes = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract function content
                func_content = self._extract_function_content(node)
                func_hash = hashlib.md5(func_content.encode()).hexdigest()
                
                functions.append({
                    "name": node.name,
                    "line": node.lineno,
                    "content": func_content,
                    "hash": func_hash,
                    "length": len(func_content.split('\n'))
                })
                
                if func_hash in function_hashes:
                    function_hashes[func_hash].append((file_path, node.lineno, node.end_lineno))
                else:
                    function_hashes[func_hash] = [(file_path, node.lineno, node.end_lineno)]
        
        # Find duplicate functions
        duplicate_functions = []
        for func_hash, locations in function_hashes.items():
            if len(locations) > 1:
                # Find the function content
                func_content = next(f["content"] for f in functions if f["hash"] == func_hash)
                
                duplicate_functions.append(DuplicationBlock(
                    content=func_content,
                    hash_value=func_hash,
                    locations=locations,
                    block_type="function"
                ))
        
        return {
            "total_functions": len(functions),
            "duplicate_functions": [f.to_dict() for f in duplicate_functions],
            "duplicate_count": len(duplicate_functions)
        }
    
    def _analyze_class_duplications(self, tree: ast.AST, file_path: str) -> Dict[str, Any]:
        """Analyze class-level duplications."""
        classes = []
        class_hashes = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Extract class content
                class_content = self._extract_class_content(node)
                class_hash = hashlib.md5(class_content.encode()).hexdigest()
                
                classes.append({
                    "name": node.name,
                    "line": node.lineno,
                    "content": class_content,
                    "hash": class_hash,
                    "method_count": len([child for child in node.body if isinstance(child, ast.FunctionDef)])
                })
                
                if class_hash in class_hashes:
                    class_hashes[class_hash].append((file_path, node.lineno, node.end_lineno))
                else:
                    class_hashes[class_hash] = [(file_path, node.lineno, node.end_lineno)]
        
        # Find duplicate classes
        duplicate_classes = []
        for class_hash, locations in class_hashes.items():
            if len(locations) > 1:
                # Find the class content
                class_content = next(c["content"] for c in classes if c["hash"] == class_hash)
                
                duplicate_classes.append(DuplicationBlock(
                    content=class_content,
                    hash_value=class_hash,
                    locations=locations,
                    block_type="class"
                ))
        
        return {
            "total_classes": len(classes),
            "duplicate_classes": [c.to_dict() for c in duplicate_classes],
            "duplicate_count": len(duplicate_classes)
        }
    
    def _analyze_code_blocks(self, tree: ast.AST, file_path: str) -> Dict[str, Any]:
        """Analyze code block duplications."""
        code_blocks = []
        block_hashes = {}
        
        # Extract code blocks (if statements, loops, etc.)
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
                block_content = self._extract_node_content(node)
                block_hash = hashlib.md5(block_content.encode()).hexdigest()
                
                code_blocks.append({
                    "type": type(node).__name__,
                    "line": node.lineno,
                    "content": block_content,
                    "hash": block_hash,
                    "length": len(block_content.split('\n'))
                })
                
                if block_hash in block_hashes:
                    block_hashes[block_hash].append((file_path, node.lineno, getattr(node, 'end_lineno', node.lineno + 1)))
                else:
                    block_hashes[block_hash] = [(file_path, node.lineno, getattr(node, 'end_lineno', node.lineno + 1))]
        
        # Find duplicate blocks
        duplicate_blocks = []
        for block_hash, locations in block_hashes.items():
            if len(locations) > 1:
                # Find the block content
                block_content = next(b["content"] for b in code_blocks if b["hash"] == block_hash)
                
                duplicate_blocks.append(DuplicationBlock(
                    content=block_content,
                    hash_value=block_hash,
                    locations=locations,
                    block_type="code_block"
                ))
        
        return {
            "total_blocks": len(code_blocks),
            "duplicate_blocks": [b.to_dict() for b in duplicate_blocks],
            "duplicate_count": len(duplicate_blocks)
        }
    
    def _analyze_code_patterns(self, tree: ast.AST, file_path: str) -> Dict[str, Any]:
        """Analyze code pattern duplications."""
        patterns = []
        pattern_hashes = {}
        
        # Extract common patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                pattern_content = self._extract_node_content(node)
                pattern_hash = hashlib.md5(pattern_content.encode()).hexdigest()
                
                patterns.append({
                    "type": "assignment",
                    "line": node.lineno,
                    "content": pattern_content,
                    "hash": pattern_hash
                })
                
                if pattern_hash in pattern_hashes:
                    pattern_hashes[pattern_hash].append((file_path, node.lineno, node.lineno))
                else:
                    pattern_hashes[pattern_hash] = [(file_path, node.lineno, node.lineno)]
            
            elif isinstance(node, ast.Return):
                pattern_content = self._extract_node_content(node)
                pattern_hash = hashlib.md5(pattern_content.encode()).hexdigest()
                
                patterns.append({
                    "type": "return",
                    "line": node.lineno,
                    "content": pattern_content,
                    "hash": pattern_hash
                })
                
                if pattern_hash in pattern_hashes:
                    pattern_hashes[pattern_hash].append((file_path, node.lineno, node.lineno))
                else:
                    pattern_hashes[pattern_hash] = [(file_path, node.lineno, node.lineno)]
        
        # Find duplicate patterns
        duplicate_patterns = []
        for pattern_hash, locations in pattern_hashes.items():
            if len(locations) > 2:  # Only consider patterns that appear more than twice
                # Find the pattern content
                pattern_content = next(p["content"] for p in patterns if p["hash"] == pattern_hash)
                
                duplicate_patterns.append(DuplicationBlock(
                    content=pattern_content,
                    hash_value=pattern_hash,
                    locations=locations,
                    block_type="code_pattern"
                ))
        
        return {
            "total_patterns": len(patterns),
            "duplicate_patterns": [p.to_dict() for p in duplicate_patterns],
            "duplicate_count": len(duplicate_patterns)
        }
    
    def _extract_function_content(self, node: ast.FunctionDef) -> str:
        """Extract the content of a function."""
        try:
            # Get the source lines
            lines = []
            for child in node.body:
                if hasattr(child, 'lineno') and hasattr(child, 'end_lineno'):
                    lines.append(f"Line {child.lineno}-{child.end_lineno}")
                else:
                    lines.append(f"Line {child.lineno}")
            
            # Create a simplified representation
            content = f"def {node.name}({len(node.args.args)} args):\n"
            content += f"  {len(node.body)} statements\n"
            content += f"  Decorators: {len(node.decorator_list)}\n"
            content += f"  Returns: {node.returns is not None}\n"
            
            return content
        except Exception:
            return f"def {node.name}()"
    
    def _extract_class_content(self, node: ast.ClassDef) -> str:
        """Extract the content of a class."""
        try:
            methods = [child for child in node.body if isinstance(child, ast.FunctionDef)]
            attributes = [child for child in node.body if isinstance(child, ast.Assign)]
            
            content = f"class {node.name}({len(node.bases)} bases):\n"
            content += f"  Methods: {len(methods)}\n"
            content += f"  Attributes: {len(attributes)}\n"
            content += f"  Decorators: {len(node.decorator_list)}\n"
            
            return content
        except Exception:
            return f"class {node.name}"
    
    def _extract_node_content(self, node: ast.AST) -> str:
        """Extract the content of an AST node."""
        try:
            if isinstance(node, ast.If):
                return f"if statement with {len(node.body)} body statements"
            elif isinstance(node, ast.While):
                return f"while loop with {len(node.body)} body statements"
            elif isinstance(node, ast.For):
                return f"for loop with {len(node.body)} body statements"
            elif isinstance(node, ast.Try):
                return f"try block with {len(node.body)} body statements"
            elif isinstance(node, ast.Assign):
                return f"assignment with {len(node.targets)} targets"
            elif isinstance(node, ast.Return):
                return f"return statement"
            else:
                return f"{type(node).__name__} node"
        except Exception:
            return f"{type(node).__name__}"
    
    def _calculate_duplication_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate duplication score (lower is better)."""
        score = 100.0
        
        # Penalize based on duplication counts
        function_dups = analysis_results.get("function_duplications", {}).get("duplicate_count", 0)
        class_dups = analysis_results.get("class_duplications", {}).get("duplicate_count", 0)
        block_dups = analysis_results.get("block_duplications", {}).get("duplicate_count", 0)
        pattern_dups = analysis_results.get("pattern_duplications", {}).get("duplicate_count", 0)
        
        total_dups = function_dups + class_dups + block_dups + pattern_dups
        
        # Apply penalties
        score -= function_dups * 15  # Function duplication is most serious
        score -= class_dups * 12     # Class duplication is serious
        score -= block_dups * 8      # Block duplication is moderate
        score -= pattern_dups * 5    # Pattern duplication is least serious
        
        return max(0.0, score)
    
    def generate_refactoring_suggestions(self) -> List[str]:
        """Generate refactoring suggestions based on duplication analysis."""
        suggestions = []
        
        # Analyze function duplications
        function_dups = [block for block in self.duplication_blocks if block.block_type == "function"]
        if function_dups:
            suggestions.append(f"Consider extracting {len(function_dups)} duplicate functions into shared utilities")
        
        # Analyze class duplications
        class_dups = [block for block in self.duplication_blocks if block.block_type == "class"]
        if class_dups:
            suggestions.append(f"Consider creating base classes for {len(class_dups)} duplicate classes")
        
        # Analyze block duplications
        block_dups = [block for block in self.duplication_blocks if block.block_type == "code_block"]
        if block_dups:
            suggestions.append(f"Consider extracting {len(block_dups)} duplicate code blocks into helper functions")
        
        # Analyze pattern duplications
        pattern_dups = [block for block in self.duplication_blocks if block.block_type == "code_pattern"]
        if pattern_dups:
            suggestions.append(f"Consider creating utility functions for {len(pattern_dups)} repeated code patterns")
        
        return suggestions
    
    def analyze_directory(self, directory: str) -> Dict[str, Any]:
        """
        Analyze code duplication for all Python files in a directory.
        
        Args:
            directory: Directory containing Python files to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        python_files = find_python_files(directory, self.config.analysis.exclude_patterns)
        print(f"Analyzing code duplication for {len(python_files)} Python files...")
        
        # Clear previous results
        self.duplication_blocks.clear()
        self.file_duplications.clear()
        self.duplication_metrics.clear()
        self.refactoring_suggestions.clear()
        
        total_issues = 0
        total_duplication_score = 0.0
        successful_files = 0
        
        for file_path in python_files:
            try:
                result = self.analyze_file(str(file_path))
                if result["status"] == "success":
                    total_issues += result["issues_found"]
                    total_duplication_score += result["duplication_score"]
                    successful_files += 1
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
        
        avg_duplication_score = total_duplication_score / successful_files if successful_files > 0 else 0.0
        
        # Generate refactoring suggestions
        self.refactoring_suggestions = self.generate_refactoring_suggestions()
        
        return {
            "status": "success",
            "total_files": len(python_files),
            "successful_files": successful_files,
            "total_issues": total_issues,
            "average_duplication_score": avg_duplication_score,
            "refactoring_suggestions": self.refactoring_suggestions,
            "file_stats": self.file_duplications
        }
    
    def can_analyze(self, file_path: str) -> bool:
        """Check if this analyzer can analyze the given file."""
        return file_path.endswith('.py')
    
    def analyze(self, file_path: str) -> Dict[str, Any]:
        """Analyze the given file (alias for analyze_file)."""
        return self.analyze_file(file_path)