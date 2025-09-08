#!/usr/bin/env python3
"""
Framework Detector

Detects web frameworks, libraries, and development patterns in Python codebases
to provide context-aware dead code analysis. Supports:

- Django (models, views, urls, settings)
- Flask (routes, blueprints, extensions)
- FastAPI (routers, dependencies, middleware)
- Pyramid (views, routes, configuration)
- Tornado (handlers, applications)
- Celery (tasks, workers)
- SQLAlchemy (models, sessions)
- Pytest (fixtures, tests)
- And more...
"""

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import logging


@dataclass
class FrameworkContext:
    """Context information about detected frameworks and patterns."""
    framework_name: str
    version: Optional[str] = None
    confidence: float = 0.0
    patterns_found: List[str] = field(default_factory=list)
    files_involved: List[str] = field(default_factory=list)
    configuration_files: List[str] = field(default_factory=list)
    entry_points: List[str] = field(default_factory=list)
    special_functions: List[str] = field(default_factory=list)
    special_classes: List[str] = field(default_factory=list)


@dataclass
class ProjectContext:
    """Complete project context including all detected frameworks."""
    project_type: str = "generic"
    frameworks: List[FrameworkContext] = field(default_factory=list)
    primary_framework: Optional[FrameworkContext] = None
    development_patterns: List[str] = field(default_factory=list)
    build_tools: List[str] = field(default_factory=list)
    testing_frameworks: List[str] = field(default_factory=list)
    database_orms: List[str] = field(default_factory=list)
    web_servers: List[str] = field(default_factory=list)
    task_queues: List[str] = field(default_factory=list)


class FrameworkDetector:
    """Detects frameworks and development patterns in Python projects."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.framework_patterns = self._initialize_framework_patterns()
        self.import_patterns = self._initialize_import_patterns()
        self.file_patterns = self._initialize_file_patterns()
        self.config_patterns = self._initialize_config_patterns()
    
    def detect_frameworks(self, project_root: Path) -> ProjectContext:
        """Detect all frameworks and patterns in the project."""
        self.logger.info(f"Detecting frameworks in {project_root}")
        
        context = ProjectContext()
        
        # Detect frameworks through various methods
        context.frameworks = self._detect_frameworks_from_imports(project_root)
        context.frameworks.extend(self._detect_frameworks_from_files(project_root))
        context.frameworks.extend(self._detect_frameworks_from_configs(project_root))
        
        # Remove duplicates and merge similar frameworks
        context.frameworks = self._merge_framework_contexts(context.frameworks)
        
        # Determine primary framework
        if context.frameworks:
            context.primary_framework = max(context.frameworks, key=lambda f: f.confidence)
            context.project_type = context.primary_framework.framework_name
        
        # Detect additional patterns
        context.development_patterns = self._detect_development_patterns(project_root)
        context.build_tools = self._detect_build_tools(project_root)
        context.testing_frameworks = self._detect_testing_frameworks(project_root)
        context.database_orms = self._detect_database_orms(project_root)
        context.web_servers = self._detect_web_servers(project_root)
        context.task_queues = self._detect_task_queues(project_root)
        
        self.logger.info(f"Detected {len(context.frameworks)} frameworks")
        return context
    
    def _initialize_framework_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize framework detection patterns."""
        return {
            "django": {
                "imports": ["django", "django.db", "django.http", "django.views", "django.urls"],
                "classes": ["Model", "View", "Form", "Admin", "Middleware"],
                "functions": ["render", "redirect", "get_object_or_404"],
                "decorators": ["@login_required", "@csrf_exempt", "@require_http_methods"],
                "files": ["models.py", "views.py", "urls.py", "settings.py", "admin.py"],
                "config_files": ["settings.py", "urls.py", "wsgi.py", "manage.py"],
                "entry_points": ["manage.py", "wsgi.py", "asgi.py"]
            },
            "flask": {
                "imports": ["flask", "flask_sqlalchemy", "flask_migrate", "flask_login"],
                "classes": ["Flask", "Blueprint", "SQLAlchemy", "Migrate"],
                "functions": ["render_template", "redirect", "url_for", "flash"],
                "decorators": ["@app.route", "@blueprint.route", "@login_required"],
                "files": ["app.py", "routes.py", "models.py", "config.py"],
                "config_files": ["config.py", "app.py"],
                "entry_points": ["app.py", "run.py", "wsgi.py"]
            },
            "fastapi": {
                "imports": ["fastapi", "uvicorn", "pydantic", "sqlalchemy"],
                "classes": ["FastAPI", "APIRouter", "BaseModel", "Depends"],
                "functions": ["get", "post", "put", "delete", "Depends"],
                "decorators": ["@app.get", "@app.post", "@router.get", "@router.post"],
                "files": ["main.py", "routers.py", "models.py", "dependencies.py"],
                "config_files": ["main.py"],
                "entry_points": ["main.py", "app.py"]
            },
            "pyramid": {
                "imports": ["pyramid", "pyramid.view", "pyramid.config", "pyramid.response"],
                "classes": ["Configurator", "View", "Response"],
                "functions": ["view_config", "render_to_response"],
                "decorators": ["@view_config"],
                "files": ["views.py", "models.py", "__init__.py"],
                "config_files": ["__init__.py", "development.ini", "production.ini"],
                "entry_points": ["__init__.py", "main.py"]
            },
            "tornado": {
                "imports": ["tornado", "tornado.web", "tornado.ioloop", "tornado.httpserver"],
                "classes": ["Application", "RequestHandler", "IOLoop"],
                "functions": ["get", "post", "write", "redirect"],
                "decorators": [],
                "files": ["main.py", "handlers.py"],
                "config_files": ["main.py"],
                "entry_points": ["main.py"]
            },
            "celery": {
                "imports": ["celery", "celery.task", "celery.worker"],
                "classes": ["Celery", "Task"],
                "functions": ["task", "delay", "apply_async"],
                "decorators": ["@task", "@periodic_task"],
                "files": ["tasks.py", "celery_app.py"],
                "config_files": ["celeryconfig.py", "celery.py"],
                "entry_points": ["celery_app.py", "tasks.py"]
            },
            "sqlalchemy": {
                "imports": ["sqlalchemy", "sqlalchemy.orm", "sqlalchemy.ext.declarative"],
                "classes": ["Base", "Column", "Integer", "String", "ForeignKey"],
                "functions": ["create_engine", "sessionmaker", "relationship"],
                "decorators": [],
                "files": ["models.py", "database.py"],
                "config_files": ["database.py", "models.py"],
                "entry_points": []
            },
            "pytest": {
                "imports": ["pytest", "pytest.fixture", "pytest.mark"],
                "classes": [],
                "functions": ["fixture", "mark", "parametrize"],
                "decorators": ["@pytest.fixture", "@pytest.mark.parametrize"],
                "files": ["test_*.py", "conftest.py"],
                "config_files": ["pytest.ini", "conftest.py", "pyproject.toml"],
                "entry_points": ["conftest.py"]
            }
        }
    
    def _initialize_import_patterns(self) -> Dict[str, List[str]]:
        """Initialize import-based detection patterns."""
        return {
            "django": ["django", "django.db", "django.http", "django.views", "django.urls", "django.contrib"],
            "flask": ["flask", "flask_sqlalchemy", "flask_migrate", "flask_login", "flask_wtf"],
            "fastapi": ["fastapi", "uvicorn", "pydantic"],
            "pyramid": ["pyramid", "pyramid.view", "pyramid.config"],
            "tornado": ["tornado", "tornado.web", "tornado.ioloop"],
            "celery": ["celery", "celery.task", "celery.worker"],
            "sqlalchemy": ["sqlalchemy", "sqlalchemy.orm"],
            "pytest": ["pytest", "pytest.fixture"],
            "numpy": ["numpy", "np"],
            "pandas": ["pandas", "pd"],
            "matplotlib": ["matplotlib", "plt"],
            "requests": ["requests"],
            "beautifulsoup4": ["bs4", "beautifulsoup4"],
            "scrapy": ["scrapy"],
            "selenium": ["selenium"],
            "tensorflow": ["tensorflow", "tf"],
            "torch": ["torch"],
            "sklearn": ["sklearn", "sklearn.model_selection"],
            "opencv": ["cv2"],
            "pillow": ["PIL"],
            "jinja2": ["jinja2"],
            "werkzeug": ["werkzeug"],
            "gunicorn": ["gunicorn"],
            "uwsgi": ["uwsgi"]
        }
    
    def _initialize_file_patterns(self) -> Dict[str, List[str]]:
        """Initialize file-based detection patterns."""
        return {
            "django": ["manage.py", "settings.py", "urls.py", "wsgi.py", "models.py", "views.py", "admin.py"],
            "flask": ["app.py", "run.py", "config.py", "requirements.txt"],
            "fastapi": ["main.py", "app.py", "requirements.txt"],
            "pyramid": ["__init__.py", "development.ini", "production.ini"],
            "tornado": ["main.py", "handlers.py"],
            "celery": ["celery_app.py", "tasks.py", "celeryconfig.py"],
            "pytest": ["conftest.py", "pytest.ini", "test_*.py"],
            "setuptools": ["setup.py", "setup.cfg", "pyproject.toml"],
            "poetry": ["pyproject.toml", "poetry.lock"],
            "pipenv": ["Pipfile", "Pipfile.lock"],
            "conda": ["environment.yml", "conda.yml"],
            "docker": ["Dockerfile", "docker-compose.yml"],
            "kubernetes": ["k8s.yml", "deployment.yaml", "service.yaml"],
            "terraform": ["*.tf", "*.tfvars"],
            "ansible": ["playbook.yml", "inventory.ini"]
        }
    
    def _initialize_config_patterns(self) -> Dict[str, List[str]]:
        """Initialize configuration-based detection patterns."""
        return {
            "django": ["DEBUG", "SECRET_KEY", "DATABASES", "INSTALLED_APPS"],
            "flask": ["SECRET_KEY", "SQLALCHEMY_DATABASE_URI", "DEBUG"],
            "fastapi": ["title", "version", "description"],
            "celery": ["CELERY_BROKER_URL", "CELERY_RESULT_BACKEND"],
            "pytest": ["testpaths", "python_files", "python_classes"],
            "docker": ["FROM", "RUN", "COPY", "WORKDIR"],
            "kubernetes": ["apiVersion", "kind", "metadata", "spec"],
            "terraform": ["provider", "resource", "variable", "output"]
        }
    
    def _detect_frameworks_from_imports(self, project_root: Path) -> List[FrameworkContext]:
        """Detect frameworks by analyzing import statements."""
        frameworks = []
        import_counts = defaultdict(int)
        framework_files = defaultdict(set)
        
        for py_file in project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            module_name = alias.name.split('.')[0]
                            if module_name in self.import_patterns:
                                for framework, imports in self.import_patterns.items():
                                    if module_name in imports:
                                        import_counts[framework] += 1
                                        framework_files[framework].add(str(py_file))
                    
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            module_name = node.module.split('.')[0]
                            if module_name in self.import_patterns:
                                for framework, imports in self.import_patterns.items():
                                    if module_name in imports:
                                        import_counts[framework] += 1
                                        framework_files[framework].add(str(py_file))
            
            except Exception as e:
                self.logger.warning(f"Failed to parse {py_file}: {e}")
        
        # Create framework contexts based on import analysis
        for framework, count in import_counts.items():
            if count > 0:
                confidence = min(count / 10.0, 1.0)  # Normalize confidence
                
                frameworks.append(FrameworkContext(
                    framework_name=framework,
                    confidence=confidence,
                    patterns_found=[f"imports_{count}"],
                    files_involved=list(framework_files[framework])
                ))
        
        return frameworks
    
    def _detect_frameworks_from_files(self, project_root: Path) -> List[FrameworkContext]:
        """Detect frameworks by analyzing file structure."""
        frameworks = []
        file_counts = defaultdict(int)
        framework_files = defaultdict(set)
        
        for framework, file_patterns in self.file_patterns.items():
            for pattern in file_patterns:
                if '*' in pattern:
                    # Handle glob patterns
                    for file_path in project_root.rglob(pattern):
                        file_counts[framework] += 1
                        framework_files[framework].add(str(file_path))
                else:
                    # Handle exact file names
                    for file_path in project_root.rglob(pattern):
                        file_counts[framework] += 1
                        framework_files[framework].add(str(file_path))
        
        # Create framework contexts based on file analysis
        for framework, count in file_counts.items():
            if count > 0:
                confidence = min(count / 5.0, 1.0)  # Normalize confidence
                
                frameworks.append(FrameworkContext(
                    framework_name=framework,
                    confidence=confidence,
                    patterns_found=[f"files_{count}"],
                    files_involved=list(framework_files[framework])
                ))
        
        return frameworks
    
    def _detect_frameworks_from_configs(self, project_root: Path) -> List[FrameworkContext]:
        """Detect frameworks by analyzing configuration files."""
        frameworks = []
        
        for framework, config_patterns in self.config_patterns.items():
            config_files = []
            patterns_found = []
            
            for py_file in project_root.rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    found_patterns = []
                    for pattern in config_patterns:
                        if pattern in content:
                            found_patterns.append(pattern)
                    
                    if found_patterns:
                        config_files.append(str(py_file))
                        patterns_found.extend(found_patterns)
                
                except Exception as e:
                    self.logger.warning(f"Failed to read {py_file}: {e}")
            
            if config_files:
                confidence = min(len(patterns_found) / 5.0, 1.0)
                
                frameworks.append(FrameworkContext(
                    framework_name=framework,
                    confidence=confidence,
                    patterns_found=patterns_found,
                    files_involved=config_files,
                    configuration_files=config_files
                ))
        
        return frameworks
    
    def _merge_framework_contexts(self, frameworks: List[FrameworkContext]) -> List[FrameworkContext]:
        """Merge duplicate framework contexts."""
        merged = {}
        
        for framework in frameworks:
            name = framework.framework_name
            
            if name in merged:
                # Merge with existing framework
                existing = merged[name]
                existing.confidence = max(existing.confidence, framework.confidence)
                existing.patterns_found.extend(framework.patterns_found)
                existing.files_involved.extend(framework.files_involved)
                existing.configuration_files.extend(framework.configuration_files)
                existing.entry_points.extend(framework.entry_points)
                existing.special_functions.extend(framework.special_functions)
                existing.special_classes.extend(framework.special_classes)
                
                # Remove duplicates
                existing.patterns_found = list(set(existing.patterns_found))
                existing.files_involved = list(set(existing.files_involved))
                existing.configuration_files = list(set(existing.configuration_files))
                existing.entry_points = list(set(existing.entry_points))
                existing.special_functions = list(set(existing.special_functions))
                existing.special_classes = list(set(existing.special_classes))
            else:
                merged[name] = framework
        
        return list(merged.values())
    
    def _detect_development_patterns(self, project_root: Path) -> List[str]:
        """Detect development patterns and practices."""
        patterns = []
        
        # Check for common development patterns
        if (project_root / "tests").exists():
            patterns.append("test_driven_development")
        
        if (project_root / "docs").exists():
            patterns.append("documentation_driven")
        
        if (project_root / ".github").exists():
            patterns.append("github_workflows")
        
        if (project_root / "scripts").exists():
            patterns.append("automation_scripts")
        
        if (project_root / "migrations").exists():
            patterns.append("database_migrations")
        
        if (project_root / "static").exists():
            patterns.append("static_assets")
        
        if (project_root / "templates").exists():
            patterns.append("template_engine")
        
        return patterns
    
    def _detect_build_tools(self, project_root: Path) -> List[str]:
        """Detect build and dependency management tools."""
        tools = []
        
        if (project_root / "setup.py").exists():
            tools.append("setuptools")
        
        if (project_root / "pyproject.toml").exists():
            tools.append("poetry")
        
        if (project_root / "Pipfile").exists():
            tools.append("pipenv")
        
        if (project_root / "requirements.txt").exists():
            tools.append("pip")
        
        if (project_root / "environment.yml").exists():
            tools.append("conda")
        
        if (project_root / "Makefile").exists():
            tools.append("make")
        
        return tools
    
    def _detect_testing_frameworks(self, project_root: Path) -> List[str]:
        """Detect testing frameworks."""
        frameworks = []
        
        # Check for test files and configurations
        test_files = list(project_root.rglob("test_*.py"))
        if test_files:
            frameworks.append("pytest")
        
        if (project_root / "conftest.py").exists():
            frameworks.append("pytest")
        
        if (project_root / "unittest").exists():
            frameworks.append("unittest")
        
        # Check imports in test files
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if "import unittest" in content:
                    frameworks.append("unittest")
                if "import pytest" in content:
                    frameworks.append("pytest")
                if "import nose" in content:
                    frameworks.append("nose")
            
            except Exception:
                pass
        
        return list(set(frameworks))
    
    def _detect_database_orms(self, project_root: Path) -> List[str]:
        """Detect database ORMs and database tools."""
        orms = []
        
        # Check for ORM imports
        for py_file in project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if "import sqlalchemy" in content or "from sqlalchemy" in content:
                    orms.append("sqlalchemy")
                if "import django.db" in content or "from django.db" in content:
                    orms.append("django_orm")
                if "import peewee" in content or "from peewee" in content:
                    orms.append("peewee")
                if "import sqlmodel" in content or "from sqlmodel" in content:
                    orms.append("sqlmodel")
            
            except Exception:
                pass
        
        return list(set(orms))
    
    def _detect_web_servers(self, project_root: Path) -> List[str]:
        """Detect web servers and deployment tools."""
        servers = []
        
        # Check for server configurations
        for py_file in project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if "import gunicorn" in content or "from gunicorn" in content:
                    servers.append("gunicorn")
                if "import uwsgi" in content or "from uwsgi" in content:
                    servers.append("uwsgi")
                if "import uvicorn" in content or "from uvicorn" in content:
                    servers.append("uvicorn")
                if "import waitress" in content or "from waitress" in content:
                    servers.append("waitress")
            
            except Exception:
                pass
        
        # Check for Docker files
        if (project_root / "Dockerfile").exists():
            servers.append("docker")
        
        if (project_root / "docker-compose.yml").exists():
            servers.append("docker_compose")
        
        return list(set(servers))
    
    def _detect_task_queues(self, project_root: Path) -> List[str]:
        """Detect task queue systems."""
        queues = []
        
        for py_file in project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if "import celery" in content or "from celery" in content:
                    queues.append("celery")
                if "import rq" in content or "from rq" in content:
                    queues.append("rq")
                if "import dramatiq" in content or "from dramatiq" in content:
                    queues.append("dramatiq")
                if "import huey" in content or "from huey" in content:
                    queues.append("huey")
            
            except Exception:
                pass
        
        return list(set(queues))


def main():
    """Main entry point for testing the framework detector."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Framework Detector")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    parser.add_argument("--output", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = FrameworkDetector()
    
    # Detect frameworks
    project_root = Path(args.project_root)
    context = detector.detect_frameworks(project_root)
    
    # Print results
    print(f"\nFramework Detection Results for {project_root}:")
    print(f"Project type: {context.project_type}")
    print(f"Primary framework: {context.primary_framework.framework_name if context.primary_framework else 'None'}")
    print(f"Frameworks detected: {len(context.frameworks)}")
    
    for framework in context.frameworks:
        print(f"  - {framework.framework_name}: {framework.confidence:.2f} confidence")
        print(f"    Files: {len(framework.files_involved)}")
        print(f"    Patterns: {', '.join(framework.patterns_found[:3])}")
    
    print(f"\nDevelopment patterns: {', '.join(context.development_patterns)}")
    print(f"Build tools: {', '.join(context.build_tools)}")
    print(f"Testing frameworks: {', '.join(context.testing_frameworks)}")
    print(f"Database ORMs: {', '.join(context.database_orms)}")
    print(f"Web servers: {', '.join(context.web_servers)}")
    print(f"Task queues: {', '.join(context.task_queues)}")
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(context.__dict__, f, indent=2, default=str)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()