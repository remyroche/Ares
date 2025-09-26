#!/usr/bin/env python3
"""Utility script to scan the repository for common code issues.

The script performs three targeted checks:

1. **Import issues** – identifies import statements that cannot be resolved
   using Python's module resolution rules.
2. **Function parameter and call issues** – leverages the improved signature
   analyzer to flag mismatches between function definitions and their usages.
3. **Redundant/dead code** – reports functions that are defined but never
   called anywhere in the project (based on static analysis).

Results are saved to a JSON report and also summarised in the console.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_QUALITY_PATH = REPO_ROOT / "code_quality"

for path in (CODE_QUALITY_PATH, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


class SimpleAnalysisConfig:
    """Minimal analysis configuration used for scanning."""

    def __init__(self) -> None:
        self.exclude_patterns = ["*.pyc", "*.pyo", "__pycache__"]
        self.exclude_directories = [
            "venv",
            "__pycache__",
            ".git",
            "node_modules",
            ".pytest_cache",
        ]


class SimpleCodeQualityConfig:
    """Minimal configuration compatible with the analyzers used."""

    def __init__(self) -> None:
        self.analysis = SimpleAnalysisConfig()


def _register_core_config_stub() -> None:
    """Install a lightweight ``core.config`` module for analyzer imports."""

    analysis_cls = SimpleAnalysisConfig

    class _CodeQualityConfig:
        def __init__(self) -> None:
            self.analysis = analysis_cls()

    core_module = types.ModuleType("core")
    core_config_module = types.ModuleType("core.config")
    core_config_module.AnalysisConfig = analysis_cls
    core_config_module.CodeQualityConfig = _CodeQualityConfig
    core_module.config = core_config_module  # type: ignore[attr-defined]

    if "core" not in sys.modules:
        sys.modules["core"] = core_module
    sys.modules["core.config"] = core_config_module


_register_core_config_stub()

from analyzers.import_analyzer import ImportAnalyzer
from analyzers.improved_signature_analyzer import ImprovedSignatureAnalyzer
from src.utils.tprint import tprint


def _normalise_module_name(module: str) -> str:
    """Return the base module name for resolution checks."""
    return module.split(" as ", 1)[0].split(".", 1)[0]


def _find_repo_root(start: Path) -> Path:
    """Return the repository root by looking for common markers."""
    current = start.resolve()
    markers = {".git", "pyproject.toml", "setup.cfg", "setup.py"}

    while True:
        if any((current / marker).exists() for marker in markers):
            return current
        if current.parent == current:
            return start.resolve()
        current = current.parent


def _detect_missing_modules(
    import_results: Dict[str, Any], project_root: Path
) -> list[dict[str, Any]]:
    """Detect import statements that cannot be resolved."""
    missing: list[dict[str, Any]] = []
    if "files" not in import_results:
        return missing

    repo_root = _find_repo_root(project_root)
    search_roots = [
        project_root,
        repo_root,
        repo_root / "src",
        repo_root / "code_quality",
    ]
    local_prefixes: set[str] = set()
    for root in search_roots:
        if root.is_dir():
            local_prefixes.update(
                child.name for child in root.iterdir() if child.is_dir()
            )

    for file_path, file_info in import_results["files"].items():
        for entry in file_info.get("imports", []):
            module = entry.get("module")
            if not module:
                continue
            if module.startswith("."):
                # Relative import – resolution requires full package context, skip.
                continue

            normalised = _normalise_module_name(module)
            if normalised in sys.builtin_module_names:
                continue
            if normalised not in local_prefixes:
                # Assume third-party or optional dependency; skip heavy checks.
                continue

            parts = module.split(".")
            found = False
            for root in search_roots:
                if not root.is_dir():
                    continue
                module_dir = root.joinpath(*parts[:-1]) if len(parts) > 1 else root
                file_candidate = module_dir / f"{parts[-1]}.py"
                package_candidate = root.joinpath(*parts, "__init__.py")

                if file_candidate.exists() or package_candidate.exists():
                    found = True
                    break

            if not found:
                missing.append(
                    {
                        "file": file_path,
                        "module": module,
                        "import_type": entry.get("type", "unknown"),
                    }
                )

    return missing


def _summarise_unused_modules(import_results: Dict[str, Any]) -> list[dict[str, Any]]:
    """Highlight modules that are never imported by other files."""
    # This uses a simple heuristic: modules with zero import count.
    unused_modules: list[dict[str, Any]] = []
    if "files" not in import_results:
        return unused_modules

    for file_path, file_info in import_results["files"].items():
        if file_info.get("import_count", 0) == 0:
            unused_modules.append({"file": file_path})

    return unused_modules


def _extract_signature_issues(
    signature_results: Dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Return only the relevant issue categories from signature analysis."""
    issues = signature_results.get("issues", {})
    return {
        "compatibility_issues": issues.get("compatibility_issues", []),
        "unused_functions": issues.get("unused_functions", []),
        "missing_functions": issues.get("missing_functions", []),
        "signature_changes": issues.get("signature_changes", []),
    }


def run_scan(project_root: Path, output_path: Path | None) -> dict[str, Any]:
    """Run all scans and return a consolidated report."""
    tprint(f"Running code issue scan for {project_root}")

    config = SimpleCodeQualityConfig()

    import_analyzer = ImportAnalyzer(config)
    import_results = import_analyzer.analyze_directory(str(project_root))
    missing_modules = _detect_missing_modules(import_results, project_root)
    unused_modules = _summarise_unused_modules(import_results)

    signature_analyzer = ImprovedSignatureAnalyzer(config)
    signature_results = signature_analyzer.analyze_directory(str(project_root))
    signature_issues = _extract_signature_issues(signature_results)

    report: dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "project_root": str(project_root),
        },
        "import_analysis": {
            "summary": {
                "total_files": len(import_results.get("files", {})),
                "total_imports": import_results.get("total_imports", 0),
            },
            "missing_modules": missing_modules,
            "unused_modules": unused_modules,
        },
        "function_analysis": {
            "summary": signature_results.get("summary", {}),
            "issues": signature_issues,
        },
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2)
        tprint(f"Report written to {output_path}")

    tprint("\nScan Summary:")
    tprint(f" - Missing modules: {len(missing_modules)}")
    tprint(f" - Files with no imports: {len(unused_modules)}")
    tprint(
        " - Function compatibility issues: "
        f"{len(signature_issues['compatibility_issues'])}"
    )
    tprint(f" - Unused functions: {len(signature_issues['unused_functions'])}")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan the repository for common code issues")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Root directory of the project to analyse",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_scan(args.project_root, args.output)


if __name__ == "__main__":
    main()
