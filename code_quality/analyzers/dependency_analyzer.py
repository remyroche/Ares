"""
Dependency Analyzer - Analyzes Python package dependencies, imports, and external library usage.
"""

import ast
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pkg_resources

from core.config import CodeQualityConfig, get_default_config
from utils.file_utils import find_python_files


class DependencyInfo:
    """Container for dependency information."""

    def __init__(self, name: str, version: str = "", source: str = "",
                 is_installed: bool = False, is_used: bool = False):
        self.name = name
        self.version = version
        self.source = source  # 'requirements.txt', 'setup.py', 'imports', etc.
        self.is_installed = is_installed
        self.is_used = is_used
        self.import_locations: list[str] = []
        self.usage_count = 0

    def __repr__(self):
        return f"DependencyInfo({self.name}@{self.version}, installed={self.is_installed}, used={self.is_used})"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "is_installed": self.is_installed,
            "is_used": self.is_used,
            "import_locations": self.import_locations,
            "usage_count": self.usage_count,
        }


class DependencyAnalyzer:
    """
    Analyzes Python package dependencies, imports, and external library usage.
    """

    def __init__(self, config: CodeQualityConfig | None = None):
        self.config = config or get_default_config()
        self.dependencies: dict[str, DependencyInfo] = {}
        self.import_analysis: dict[str, dict[str, list[str]]] = {}
        self.package_usage: dict[str, set[str]] = defaultdict(set)
        self.missing_dependencies: list[str] = []
        self.unused_dependencies: list[str] = []

    def analyze_directory(self, directory: str) -> dict[str, Any]:
        """
        Analyze dependencies for all Python files in a directory.

        Args:
            directory: Directory containing Python files to analyze

        Returns:
            Dictionary containing dependency analysis results
        """
        python_files = find_python_files(directory, self.config.analysis.exclude_patterns)
        print(f"Analyzing dependencies for {len(python_files)} Python files...")

        # Clear previous results
        self.dependencies.clear()
        self.import_analysis.clear()
        self.package_usage.clear()
        self.missing_dependencies.clear()
        self.unused_dependencies.clear()

        # Analyze imports in all files
        for file_path in python_files:
            self._analyze_file_imports(file_path)

        # Find dependency files
        dependency_files = self._find_dependency_files(directory)

        # Parse dependency files
        for dep_file in dependency_files:
            self._parse_dependency_file(dep_file)

        # Check installed packages
        self._check_installed_packages()

        # Analyze usage patterns
        self._analyze_usage_patterns()

        # Generate analysis results
        return self._generate_analysis_results()


    def _analyze_file_imports(self, file_path: str) -> None:
        """Analyze imports in a single Python file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            file_imports = {
                "imports": [],
                "from_imports": [],
                "relative_imports": [],
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_name = alias.asname or alias.name
                        module_name = alias.name.split(".")[0]  # Get top-level module

                        file_imports["imports"].append({
                            "name": import_name,
                            "module": alias.name,
                            "line": node.lineno,
                        })

                        # Track package usage
                        self.package_usage[module_name].add(file_path)

                        # Add to dependencies
                        if module_name not in self.dependencies:
                            self.dependencies[module_name] = DependencyInfo(
                                name=module_name,
                                source="imports",
                            )
                        self.dependencies[module_name].import_locations.append(file_path)
                        self.dependencies[module_name].usage_count += 1

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        import_name = alias.asname or alias.name
                        module_name = module.split(".")[0] if module else ""

                        if module.startswith("."):
                            file_imports["relative_imports"].append({
                                "name": import_name,
                                "module": module,
                                "line": node.lineno,
                            })
                        else:
                            file_imports["from_imports"].append({
                                "name": import_name,
                                "module": module,
                                "line": node.lineno,
                            })

                            # Track package usage
                            if module_name:
                                self.package_usage[module_name].add(file_path)

                                # Add to dependencies
                                if module_name not in self.dependencies:
                                    self.dependencies[module_name] = DependencyInfo(
                                        name=module_name,
                                        source="imports",
                                    )
                                self.dependencies[module_name].import_locations.append(file_path)
                                self.dependencies[module_name].usage_count += 1

            self.import_analysis[file_path] = file_imports

        except Exception as e:
            print(f"Warning: Could not analyze imports in {file_path}: {e}")

    def _find_dependency_files(self, directory: str) -> list[str]:
        """Find dependency files in the directory."""
        dependency_files = []
        dependency_patterns = [
            "requirements.txt",
            "requirements-dev.txt",
            "requirements-test.txt",
            "setup.py",
            "pyproject.toml",
            "Pipfile",
            "poetry.lock",
        ]

        for pattern in dependency_patterns:
            file_path = os.path.join(directory, pattern)
            if os.path.exists(file_path):
                dependency_files.append(file_path)

        # Also check for requirements files in subdirectories
        for root, _dirs, files in os.walk(directory):
            for file in files:
                if file in dependency_patterns:
                    dependency_files.append(os.path.join(root, file))

        return dependency_files

    def _parse_dependency_file(self, file_path: str) -> None:
        """Parse a dependency file to extract package information."""
        file_name = os.path.basename(file_path)

        try:
            if file_name == "requirements.txt":
                self._parse_requirements_txt(file_path)
            elif file_name == "setup.py":
                self._parse_setup_py(file_path)
            elif file_name == "pyproject.toml":
                self._parse_pyproject_toml(file_path)
            elif file_name == "Pipfile":
                self._parse_pipfile(file_path)
            elif file_name == "poetry.lock":
                self._parse_poetry_lock(file_path)

        except Exception as e:
            print(f"Warning: Could not parse dependency file {file_path}: {e}")

    def _parse_requirements_txt(self, file_path: str) -> None:
        """Parse requirements.txt file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                for _line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith("#") and not line.startswith("-"):
                        # Parse package specification
                        package_name = line.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].split("!=")[0]
                        package_name = package_name.strip()

                        if package_name:
                            if package_name not in self.dependencies:
                                self.dependencies[package_name] = DependencyInfo(
                                    name=package_name,
                                    source="requirements.txt",
                                )
                            else:
                                self.dependencies[package_name].source = "requirements.txt"

        except Exception as e:
            print(f"Error parsing requirements.txt: {e}")

    def _parse_setup_py(self, file_path: str) -> None:
        """Parse setup.py file for dependencies."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Simple parsing for common patterns
            if "install_requires" in content:
                # Look for install_requires list
                lines = content.split("\n")
                in_requires = False
                for line in lines:
                    if "install_requires" in line and "=" in line:
                        in_requires = True
                        continue
                    if in_requires and line.strip().startswith("]"):
                        break
                    if in_requires and line.strip().startswith("'"):
                        package = line.strip().strip("',")
                        if package and not package.startswith("#"):
                            if package not in self.dependencies:
                                self.dependencies[package] = DependencyInfo(
                                    name=package,
                                    source="setup.py",
                                )
                            else:
                                self.dependencies[package].source = "setup.py"

        except Exception as e:
            print(f"Error parsing setup.py: {e}")

    def _parse_pyproject_toml(self, file_path: str) -> None:
        """Parse pyproject.toml file for dependencies."""
        try:
            import toml
            with open(file_path, encoding="utf-8") as f:
                data = toml.load(f)

            # Check for dependencies in various sections
            if "project" in data and "dependencies" in data["project"]:
                for dep in data["project"]["dependencies"]:
                    package_name = dep.split("==")[0].split(">=")[0].split("<=")[0]
                    if package_name not in self.dependencies:
                        self.dependencies[package_name] = DependencyInfo(
                            name=package_name,
                            source="pyproject.toml",
                        )
                    else:
                        self.dependencies[package_name].source = "pyproject.toml"

        except ImportError:
            print("Warning: toml package not available, skipping pyproject.toml parsing")
        except Exception as e:
            print(f"Error parsing pyproject.toml: {e}")

    def _parse_pipfile(self, file_path: str) -> None:
        """Parse Pipfile for dependencies."""
        try:
            import toml
            with open(file_path, encoding="utf-8") as f:
                data = toml.load(f)

            # Check for packages in various sections
            for section in ["packages", "dev-packages"]:
                if section in data:
                    for package_name in data[section]:
                        if package_name not in self.dependencies:
                            self.dependencies[package_name] = DependencyInfo(
                                name=package_name,
                                source="Pipfile",
                            )
                        else:
                            self.dependencies[package_name].source = "Pipfile"

        except ImportError:
            print("Warning: toml package not available, skipping Pipfile parsing")
        except Exception as e:
            print(f"Error parsing Pipfile: {e}")

    def _parse_poetry_lock(self, file_path: str) -> None:
        """Parse poetry.lock file for dependencies."""
        try:
            import toml
            with open(file_path, encoding="utf-8") as f:
                data = toml.load(f)

            # Check for packages in poetry.lock
            if "package" in data:
                for package in data["package"]:
                    package_name = package.get("name", "")
                    version = package.get("version", "")
                    if package_name:
                        if package_name not in self.dependencies:
                            self.dependencies[package_name] = DependencyInfo(
                                name=package_name,
                                version=version,
                                source="poetry.lock",
                            )
                        else:
                            self.dependencies[package_name].version = version
                            self.dependencies[package_name].source = "poetry.lock"

        except ImportError:
            print("Warning: toml package not available, skipping poetry.lock parsing")
        except Exception as e:
            print(f"Error parsing poetry.lock: {e}")

    def _check_installed_packages(self) -> None:
        """Check which packages are actually installed."""
        try:
            # Get installed packages
            installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

            for package_name, dep_info in self.dependencies.items():
                if package_name.lower() in installed_packages:
                    dep_info.is_installed = True
                    if not dep_info.version:
                        dep_info.version = installed_packages[package_name.lower()]
                else:
                    # Try alternative names
                    for installed_name, version in installed_packages.items():
                        if installed_name.lower() == package_name.lower():
                            dep_info.is_installed = True
                            dep_info.version = version
                            break

        except Exception as e:
            print(f"Warning: Could not check installed packages: {e}")

    def _analyze_usage_patterns(self) -> None:
        """Analyze how dependencies are used across the codebase."""
        for package_name, dep_info in self.dependencies.items():
            if package_name in self.package_usage:
                dep_info.is_used = True
                dep_info.usage_count = len(self.package_usage[package_name])

        # Find missing dependencies
        for package_name, dep_info in self.dependencies.items():
            if dep_info.source in ["requirements.txt", "setup.py", "pyproject.toml", "Pipfile"] and not dep_info.is_installed:
                self.missing_dependencies.append(package_name)

        # Find unused dependencies
        for package_name, dep_info in self.dependencies.items():
            if dep_info.source in ["requirements.txt", "setup.py", "pyproject.toml", "Pipfile"] and not dep_info.is_used:
                self.unused_dependencies.append(package_name)

    def _generate_analysis_results(self) -> dict[str, Any]:
        """Generate comprehensive analysis results."""
        analysis = {
            "total_dependencies": len(self.dependencies),
            "installed_dependencies": len([d for d in self.dependencies.values() if d.is_installed]),
            "missing_dependencies": len(self.missing_dependencies),
            "unused_dependencies": len(self.unused_dependencies),
            "used_dependencies": len([d for d in self.dependencies.values() if d.is_used]),
            "dependency_details": {name: dep.to_dict() for name, dep in self.dependencies.items()},
            "missing_dependencies_list": self.missing_dependencies,
            "unused_dependencies_list": self.unused_dependencies,
            "import_analysis": self.import_analysis,
            "package_usage": {name: list(locations) for name, locations in self.package_usage.items()},
            "dependency_sources": defaultdict(list),
        }

        # Group dependencies by source
        for name, dep in self.dependencies.items():
            analysis["dependency_sources"][dep.source].append(name)

        # Convert defaultdict to regular dict
        analysis["dependency_sources"] = dict(analysis["dependency_sources"])

        return analysis

    def get_package_dependencies(self, package_name: str) -> DependencyInfo | None:
        """Get dependency information for a specific package."""
        return self.dependencies.get(package_name)

    def find_package_usage(self, package_name: str) -> list[str]:
        """Find all files that use a specific package."""
        return list(self.package_usage.get(package_name, []))

    def get_missing_dependencies(self) -> list[str]:
        """Get list of missing dependencies."""
        return self.missing_dependencies

    def get_unused_dependencies(self) -> list[str]:
        """Get list of unused dependencies."""
        return self.unused_dependencies

    def generate_requirements_txt(self, output_path: str, include_versions: bool = True) -> None:
        """Generate a requirements.txt file from the analysis."""
        try:
            with open(output_path, "w") as f:
                f.write("# Generated requirements.txt from dependency analysis\n")
                f.write("# Only includes actually used dependencies\n\n")

                # Sort by name
                sorted_deps = sorted(self.dependencies.items(), key=lambda x: x[0].lower())

                for package_name, dep_info in sorted_deps:
                    if dep_info.is_used and dep_info.source != "imports":
                        if include_versions and dep_info.version:
                            f.write(f"{package_name}=={dep_info.version}\n")
                        else:
                            f.write(f"{package_name}\n")

            print(f"Requirements file generated: {output_path}")

        except Exception as e:
            print(f"Error generating requirements.txt: {e}")

    def check_security_vulnerabilities(self) -> list[dict[str, Any]]:
        """Check for known security vulnerabilities in dependencies."""
        vulnerabilities = []

        try:
            # Try to use safety if available
            import subprocess
            result = subprocess.run([sys.executable, "-m", "safety", "check", "--json"],
                                 check=False, capture_output=True, text=True)

            if result.returncode == 0:
                try:
                    vuln_data = json.loads(result.stdout)
                    for vuln in vuln_data:
                        vulnerabilities.append({
                            "package": vuln.get("package", ""),
                            "vulnerability_id": vuln.get("vulnerability_id", ""),
                            "severity": vuln.get("severity", ""),
                            "description": vuln.get("description", ""),
                            "affected_versions": vuln.get("affected_versions", []),
                        })
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            print(f"Warning: Could not check security vulnerabilities: {e}")

        return vulnerabilities

    def export_analysis(self, output_path: str) -> None:
        """Export the complete dependency analysis to JSON."""
        try:
            analysis_results = self._generate_analysis_results()

            with open(output_path, "w") as f:
                json.dump(analysis_results, f, indent=2)

            print(f"Dependency analysis exported to {output_path}")

        except Exception as e:
            print(f"Error exporting analysis: {e}")


def main():
    """Command-line interface for the dependency analyzer."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Python package dependencies")
    parser.add_argument("--path", required=True, help="Path to directory containing Python files")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--generate-requirements", help="Generate requirements.txt file")
    parser.add_argument("--check-security", action="store_true", help="Check for security vulnerabilities")

    args = parser.parse_args()

    # Load configuration
    if args.config:
        from core.config import load_config
        config = load_config(args.config)
    else:
        config = get_default_config()

    # Run dependency analysis
    analyzer = DependencyAnalyzer(config)
    results = analyzer.analyze_directory(args.path)

    # Print summary
    print("\n" + "="*50)
    print("DEPENDENCY ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total dependencies found: {results['total_dependencies']}")
    print(f"Installed: {results['installed_dependencies']}")
    print(f"Missing: {results['missing_dependencies']}")
    print(f"Unused: {results['unused_dependencies']}")
    print(f"Actually used: {results['used_dependencies']}")

    if results["missing_dependencies"]:
        print("\nMissing dependencies:")
        for dep in results["missing_dependencies"][:10]:
            print(f"  - {dep}")

    if results["unused_dependencies"]:
        print("\nUnused dependencies:")
        for dep in results["unused_dependencies"][:10]:
            print(f"  - {dep}")

    # Check security vulnerabilities
    if args.check_security:
        print("\nChecking security vulnerabilities...")
        vulnerabilities = analyzer.check_security_vulnerabilities()
        if vulnerabilities:
            print(f"Found {len(vulnerabilities)} security vulnerabilities:")
            for vuln in vulnerabilities[:5]:
                print(f"  - {vuln['package']}: {vuln['severity']} - {vuln['description']}")
        else:
            print("No security vulnerabilities found.")

    # Generate requirements.txt
    if args.generate_requirements:
        analyzer.generate_requirements_txt(args.generate_requirements)

    # Export results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)

        # Export analysis
        analysis_file = output_dir / "dependency_analysis.json"
        analyzer.export_analysis(str(analysis_file))
        print(f"\nAnalysis results exported to {analysis_file}")

        # Generate requirements.txt
        requirements_file = output_dir / "requirements_generated.txt"
        analyzer.generate_requirements_txt(str(requirements_file))


if __name__ == "__main__":
    main()
