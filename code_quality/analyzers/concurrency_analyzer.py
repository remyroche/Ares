"""
Concurrency Analyzer - Analyzes threading, async/await patterns, race conditions, and synchronization.
"""

import ast
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from minimal_config import CodeQualityConfig as CodeQualityConfig_minimal_config, get_default_config
from minimal_file_utils import find_python_files as find_python_files_minimal_file_utils


class ConcurrencyIssue:
    """Container for concurrency issue information."""

    def __init__(self, issue_type: str, description: str, line: int,
                 severity: str = "warning", details: dict[str, Any] = None):
        self.issue_type = issue_type
        self.description = description
        self.line = line
        self.severity = severity
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "issue_type": issue_type,
            "description": description,
            "line": line,
            "severity": severity,
            "details": details,
        }


class ConcurrencyPattern:
    """Container for concurrency pattern information."""

    def __init__(self, pattern_type: str, description: str, line: int,
                 quality_score: float, details: dict[str, Any] = None):
        self.pattern_type = pattern_type
        self.description = description
        self.line = line
        self.quality_score = quality_score
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "pattern_type": pattern_type,
            "description": description,
            "line": line,
            "quality_score": quality_score,
            "details": details,
        }


class ConcurrencyAnalyzer:
    """
    Comprehensive concurrency analysis and quality assessment.

    Features:
    - Threading pattern detection
    - Async/await pattern analysis
    - Race condition detection
    - Synchronization mechanism analysis
    - Lock and semaphore usage analysis
    - Concurrency best practice validation
    """

    def __init__(self, config: CodeQualityConfig | None = None):
        self.config = config or get_default_config()
        self.concurrency_issues: list[ConcurrencyIssue] = []
        self.concurrency_patterns: list[ConcurrencyPattern] = []
        self.concurrency_metrics: dict[str, dict[str, Any]] = {}
        self.file_stats: dict[str, dict[str, Any]] = {}

    def analyze_file(self, file_path: str) -> dict[str, Any]:
        """
        Analyze concurrency for a single Python file.

        Args:
            file_path: Path to Python file to analyze

        Returns:
            Dictionary containing analysis results
        """
        try:
            file_path = Path(file_path).resolve()

            # Clear previous results for this file
            self.concurrency_issues = [issue for issue in self.concurrency_issues if issue.description != str(file_path)]

            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            # Perform concurrency analysis
            threading_analysis = self._analyze_threading_patterns(tree)
            async_analysis = self._analyze_async_patterns(tree)
            race_condition_analysis = self._analyze_race_conditions(tree)
            synchronization_analysis = self._analyze_synchronization(tree)
            quality_analysis = self._assess_concurrency_quality(tree)

            # Combine results
            combined_results = {
                "threading": threading_analysis,
                "async_patterns": async_analysis,
                "race_conditions": race_condition_analysis,
                "synchronization": synchronization_analysis,
                "quality": quality_analysis,
            }

            # Calculate overall concurrency score
            concurrency_score = self._calculate_concurrency_score(combined_results)

            # Store results
            self.file_stats[str(file_path)] = combined_results

            return {
                "status": "success",
                "issues_found": len(self.concurrency_issues),
                "issues_fixed": 0,
                "details": combined_results,
                "concurrency_score": concurrency_score,
            }

        except Exception as e:
            logging.exception(f"Error in concurrency analysis for {file_path}: {e}")
            return {
                "status": "error",
                "issues_found": 0,
                "issues_fixed": 0,
                "details": {"error": str(e)},
                "concurrency_score": 0,
            }

    def _analyze_threading_patterns(self, tree: ast.AST) -> dict[str, Any]:
        """Analyze threading patterns and usage."""
        threading_imports = []
        thread_creations = []
        thread_joins = []
        thread_safety_issues = []

        for node in ast.walk(tree):
            # Check for threading imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ["threading", "thread", "_thread"]:
                        threading_imports.append({
                            "line": node.lineno,
                            "module": alias.name,
                        })

            elif isinstance(node, ast.ImportFrom):
                if node.module in ["threading", "thread", "_thread"]:
                    threading_imports.append({
                        "line": node.lineno,
                        "module": node.module,
                        "names": [alias.name for alias in node.names],
                    })

            # Check for Thread class usage
            elif isinstance(node, ast.ClassDef):
                if node.name == "Thread" or any("Thread" in base.id for base in node.bases if isinstance(base, ast.Name)):
                    thread_creations.append({
                        "line": node.lineno,
                        "type": "Thread_subclass",
                        "name": node.name,
                    })

            # Check for thread creation
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "Thread":
                    thread_creations.append({
                        "line": node.lineno,
                        "type": "Thread_instantiation",
                        "args": len(node.args),
                    })
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr == "start":
                        thread_creations.append({
                            "line": node.lineno,
                            "type": "thread_start",
                            "target": self._get_attribute_target(node.func.value),
                        })
                    elif node.func.attr == "join":
                        thread_joins.append({
                            "line": node.lineno,
                            "target": self._get_attribute_target(node.func.value),
                        })

        # Detect thread safety issues
        if threading_imports and not thread_creations:
            thread_safety_issues.append({
                "line": threading_imports[0]["line"],
                "type": "unused_threading_import",
                "description": "Threading module imported but not used",
            })

        return {
            "threading_imports": threading_imports,
            "thread_creations": thread_creations,
            "thread_joins": thread_joins,
            "thread_safety_issues": thread_safety_issues,
            "total_threading_operations": len(thread_creations) + len(thread_joins),
        }

    def _analyze_async_patterns(self, tree: ast.AST) -> dict[str, Any]:
        """Analyze async/await patterns."""
        async_functions = []
        await_statements = []
        async_imports = []
        event_loop_usage = []

        for node in ast.walk(tree):
            # Check for async function definitions
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith("async") or any("async" in decorator.id for decorator in node.decorator_list if isinstance(decorator, ast.Name)):
                    async_functions.append({
                        "line": node.lineno,
                        "name": node.name,
                        "is_coroutine": True,
                    })

            # Check for await statements
            elif isinstance(node, ast.Await):
                await_statements.append({
                    "line": node.lineno,
                    "value": self._get_await_value(node.value),
                })

            # Check for async imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ["asyncio", "aiohttp", "asyncpg"]:
                        async_imports.append({
                            "line": node.lineno,
                            "module": alias.name,
                        })

            elif isinstance(node, ast.ImportFrom):
                if node.module in ["asyncio", "aiohttp", "asyncpg"]:
                    async_imports.append({
                        "line": node.lineno,
                        "module": node.module,
                        "names": [alias.name for alias in node.names],
                    })

            # Check for event loop usage
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ["run", "create_task", "gather"]:
                        event_loop_usage.append({
                            "line": node.lineno,
                            "method": node.func.attr,
                            "target": self._get_attribute_target(node.func.value),
                        })

        return {
            "async_functions": async_functions,
            "await_statements": await_statements,
            "async_imports": async_imports,
            "event_loop_usage": event_loop_usage,
            "total_async_operations": len(async_functions) + len(await_statements),
        }

    def _analyze_race_conditions(self, tree: ast.AST) -> dict[str, Any]:
        """Analyze potential race conditions."""
        race_conditions = []
        shared_variables = []
        unprotected_access = []

        # Track variable access patterns
        variable_access = defaultdict(list)

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variable_access[target.id].append({
                            "line": node.lineno,
                            "type": "write",
                            "context": "assignment",
                        })

            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                variable_access[node.id].append({
                    "line": node.lineno,
                    "type": "read",
                    "context": "variable_usage",
                })

            # Check for global variable usage in functions
            elif isinstance(node, ast.FunctionDef):
                global_vars = []
                for stmt in node.body:
                    if isinstance(stmt, ast.Global):
                        global_vars.extend(stmt.names)

                if global_vars:
                    for var in global_vars:
                        if var in variable_access:
                            shared_variables.append({
                                "line": node.lineno,
                                "variable": var,
                                "function": node.name,
                                "access_count": len(variable_access[var]),
                            })

        # Detect potential race conditions
        for var, accesses in variable_access.items():
            if len(accesses) > 1:
                has_writes = any(acc["type"] == "write" for acc in accesses)
                if has_writes:
                    race_conditions.append({
                        "line": accesses[0]["line"],
                        "variable": var,
                        "type": "potential_race_condition",
                        "description": f"Variable '{var}' accessed from multiple contexts with writes",
                    })

        return {
            "race_conditions": race_conditions,
            "shared_variables": shared_variables,
            "unprotected_access": unprotected_access,
            "total_race_conditions": len(race_conditions),
        }

    def _analyze_synchronization(self, tree: ast.AST) -> dict[str, Any]:
        """Analyze synchronization mechanisms."""
        locks = []
        semaphores = []
        barriers = []
        condition_variables = []
        rlocks = []

        for node in ast.walk(tree):
            # Check for lock imports and usage
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ["threading", "multiprocessing"]:
                        locks.append({
                            "line": node.lineno,
                            "type": "lock_import",
                            "module": alias.name,
                        })

            elif isinstance(node, ast.ImportFrom):
                if node.module in ["threading", "multiprocessing"]:
                    for alias in node.names:
                        if alias.name in ["Lock", "RLock", "Semaphore", "Barrier", "Condition"]:
                            locks.append({
                                "line": node.lineno,
                                "type": f"{alias.name}_import",
                                "module": node.module,
                            })

            # Check for lock instantiation
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["Lock", "RLock", "Semaphore", "Barrier", "Condition"]:
                        locks.append({
                            "line": node.lineno,
                            "type": f"{node.func.id}_instantiation",
                            "args": len(node.args),
                        })

                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ["acquire", "release", "wait", "notify"]:
                        lock_type = self._get_lock_type(node.func.value)
                        if lock_type:
                            locks.append({
                                "line": node.lineno,
                                "type": f"{node.func.attr}_call",
                                "lock_type": lock_type,
                            })

            # Check for context manager usage (with statements)
            elif isinstance(node, ast.With):
                for item in node.items:
                    if isinstance(item.context_expr, ast.Call):
                        if isinstance(item.context_expr.func, ast.Name):
                            if item.context_expr.func.id in ["Lock", "RLock", "Semaphore"]:
                                locks.append({
                                    "line": node.lineno,
                                    "type": "context_manager_lock",
                                    "lock_type": item.context_expr.func.id,
                                })

        # Categorize locks
        for lock in locks:
            if "Lock" in lock["type"]:
                locks.append(lock)
            elif "RLock" in lock["type"]:
                rlocks.append(lock)
            elif "Semaphore" in lock["type"]:
                semaphores.append(lock)
            elif "Barrier" in lock["type"]:
                barriers.append(lock)
            elif "Condition" in lock["type"]:
                condition_variables.append(lock)

        return {
            "locks": locks,
            "semaphores": semaphores,
            "barriers": barriers,
            "condition_variables": condition_variables,
            "rlocks": rlocks,
            "total_sync_mechanisms": len(locks) + len(semaphores) + len(barriers) + len(condition_variables) + len(rlocks),
        }

    def _assess_concurrency_quality(self, tree: ast.AST) -> dict[str, Any]:
        """Assess overall concurrency quality."""
        quality_score = 100.0
        issues = []

        # Check for proper synchronization
        threading_ops = 0
        sync_mechanisms = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "Thread":
                    threading_ops += 1
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ["start", "join"]:
                        threading_ops += 1
                    elif node.func.attr in ["acquire", "release"]:
                        sync_mechanisms += 1

        # Penalize for threading without synchronization
        if threading_ops > 0 and sync_mechanisms == 0:
            quality_score -= 30
            issues.append("Threading operations without synchronization mechanisms")

        # Check for async/await best practices
        async_functions = 0
        await_statements = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("async"):
                async_functions += 1
            elif isinstance(node, ast.Await):
                await_statements += 1

        # Check for proper async/await usage
        if async_functions > 0 and await_statements == 0:
            quality_score -= 20
            issues.append("Async functions without await statements")

        return {
            "quality_score": max(0.0, quality_score),
            "issues": issues,
            "total_issues": len(issues),
            "threading_operations": threading_ops,
            "sync_mechanisms": sync_mechanisms,
            "async_functions": async_functions,
            "await_statements": await_statements,
        }

    def _get_attribute_target(self, node: ast.AST) -> str:
        """Get the target of an attribute access."""
        try:
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                return f"{self._get_attribute_target(node.value)}.{node.attr}"
            return str(node)
        except Exception:
            return "unknown"

    def _get_await_value(self, node: ast.AST) -> str:
        """Get the value being awaited."""
        try:
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    return f"{node.func.id}()"
                if isinstance(node.func, ast.Attribute):
                    return f"{self._get_attribute_target(node.func.value)}.{node.func.attr}()"
            elif isinstance(node, ast.Attribute):
                return self._get_attribute_target(node)
            else:
                return str(node)
        except Exception:
            return "unknown"

    def _get_lock_type(self, node: ast.AST) -> str | None:
        """Get the type of lock being used."""
        try:
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                return f"{self._get_attribute_target(node.value)}.{node.attr}"
            return None
        except Exception:
            return None

    def _calculate_concurrency_score(self, analysis_results: dict[str, Any]) -> float:
        """Calculate overall concurrency quality score."""
        score = 100.0

        # Quality score from analysis
        quality_score = analysis_results.get("quality", {}).get("quality_score", 100)
        score = (score + quality_score) / 2

        # Race condition penalties
        race_conditions = analysis_results.get("race_conditions", {}).get("total_race_conditions", 0)
        score -= race_conditions * 25

        # Threading without sync penalties
        threading_ops = analysis_results.get("quality", {}).get("threading_operations", 0)
        sync_mechanisms = analysis_results.get("quality", {}).get("sync_mechanisms", 0)
        if threading_ops > 0 and sync_mechanisms == 0:
            score -= 30

        # Async/await best practice penalties
        async_functions = analysis_results.get("quality", {}).get("async_functions", 0)
        await_statements = analysis_results.get("quality", {}).get("await_statements", 0)
        if async_functions > 0 and await_statements == 0:
            score -= 20

        return max(0.0, min(100.0, score))

    def analyze_directory(self, directory: str) -> dict[str, Any]:
        """
        Analyze concurrency for all Python files in a directory.

        Args:
            directory: Directory containing Python files to analyze

        Returns:
            Dictionary containing analysis results
        """
        python_files = find_python_files(directory, self.config.analysis.exclude_patterns)
        print(f"Analyzing concurrency for {len(python_files)} Python files...")

        # Clear previous results
        self.concurrency_issues.clear()
        self.concurrency_patterns.clear()
        self.concurrency_metrics.clear()
        self.file_stats.clear()

        total_issues = 0
        total_concurrency_score = 0.0
        successful_files = 0

        for file_path in python_files:
            try:
                result = self.analyze_file(str(file_path))
                if result["status"] == "success":
                    total_issues += result["issues_found"]
                    total_concurrency_score += result["concurrency_score"]
                    successful_files += 1
            except Exception as e:
                logging.exception(f"Error processing {file_path}: {e}")

        avg_concurrency_score = total_concurrency_score / successful_files if successful_files > 0 else 0.0

        return {
            "status": "success",
            "total_files": len(python_files),
            "successful_files": successful_files,
            "total_issues": total_issues,
            "average_concurrency_score": avg_concurrency_score,
            "file_stats": self.file_stats,
        }

    def can_analyze(self, file_path: str) -> bool:
        """Check if this analyzer can analyze the given file."""
        return file_path.endswith(".py")

    def analyze(self, file_path: str) -> dict[str, Any]:
        """Analyze the given file (alias for analyze_file)."""
        return self.analyze_file(file_path)
