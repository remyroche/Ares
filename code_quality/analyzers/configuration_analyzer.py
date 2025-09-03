#!/usr/bin/env python3
"""
Configuration Analyzer

Analyzes configuration quality and security including:
- Configuration file validation
- Environment variable usage
- Hardcoded credentials detection
- Configuration documentation
- Required settings validation
- Type safety in configurations
- Configuration drift detection
"""

import ast
import re
import json
import yaml
import configparser
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import os


@dataclass
class ConfigurationIssue:
    """Represents a configuration issue."""
    file_path: str
    line_number: int
    issue_type: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    message: str
    suggestion: str
    config_key: Optional[str] = None


@dataclass
class ConfigurationMetrics:
    """Metrics for configuration quality."""
    total_settings: int
    documented_settings: int
    typed_settings: int
    environment_vars: int
    hardcoded_secrets: int
    missing_required: List[str]
    unused_settings: List[str]


class ConfigurationAnalyzer:
    """Analyzes configuration quality and security."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues: List[ConfigurationIssue] = []
        self.config_files: Dict[str, Dict[str, Any]] = {}
        self.env_vars_used: Set[str] = set()
        self.config_keys_used: Set[str] = set()
        self.metrics = ConfigurationMetrics(0, 0, 0, 0, 0, [], [])
        
        # Common configuration file patterns
        self.config_patterns = [
            '**/*.yaml', '**/*.yml',
            '**/*.json',
            '**/*.ini', '**/*.cfg',
            '**/*.toml',
            '**/.env', '**/.env.*',
            '**/config.py', '**/settings.py',
            '**/configuration.py'
        ]
        
        # Sensitive key patterns
        self.sensitive_patterns = [
            r'(password|passwd|pwd)',
            r'(secret|private)',
            r'(key|token|auth)',
            r'(api_key|apikey)',
            r'(access_key|accesskey)',
            r'(credential|cred)',
            r'(certificate|cert)',
            r'(ssh|rsa|pem)'
        ]
        
        # Required configuration keys (customizable)
        self.required_keys = {
            'database': ['host', 'port', 'name'],
            'logging': ['level', 'format'],
            'security': ['secret_key', 'allowed_hosts'],
            'api': ['base_url', 'version']
        }
        
    def analyze_project(self) -> Dict[str, Any]:
        """Analyze all configuration files in the project."""
        # Find configuration files
        config_files = self._find_config_files()
        
        # Analyze each configuration file
        for config_file in config_files:
            self._analyze_config_file(config_file)
            
        # Find environment variable usage in code
        self._analyze_env_var_usage()
        
        # Find configuration usage in code
        self._analyze_config_usage()
        
        # Check for configuration issues
        self._check_configuration_issues()
        
        return self._generate_report()
        
    def _find_config_files(self) -> List[Path]:
        """Find all configuration files in the project."""
        config_files = []
        
        for pattern in self.config_patterns:
            config_files.extend(self.project_root.glob(pattern))
            
        # Filter out common non-config files
        config_files = [
            f for f in config_files
            if not any(part in str(f) for part in [
                '__pycache__', 'node_modules', '.git', 
                'venv', 'env', '.pytest_cache', 'dist'
            ])
        ]
        
        return config_files
        
    def _analyze_config_file(self, file_path: Path) -> None:
        """Analyze a single configuration file."""
        try:
            file_ext = file_path.suffix.lower()
            
            if file_ext in ['.yaml', '.yml']:
                self._analyze_yaml_config(file_path)
            elif file_ext == '.json':
                self._analyze_json_config(file_path)
            elif file_ext in ['.ini', '.cfg']:
                self._analyze_ini_config(file_path)
            elif file_ext == '.toml':
                self._analyze_toml_config(file_path)
            elif file_path.name.startswith('.env'):
                self._analyze_env_file(file_path)
            elif file_ext == '.py':
                self._analyze_python_config(file_path)
                
        except Exception as e:
            self._add_issue(
                str(file_path), 0, 'parse_error', 'high',
                f"Failed to parse configuration file: {str(e)}",
                "Check file syntax and format"
            )
            
    def _analyze_yaml_config(self, file_path: Path) -> None:
        """Analyze YAML configuration file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        try:
            config = yaml.safe_load(content)
            if config:
                self.config_files[str(file_path)] = config
                self._check_config_content(file_path, config, content)
        except yaml.YAMLError as e:
            self._add_issue(
                str(file_path), e.problem_mark.line if hasattr(e, 'problem_mark') else 0,
                'invalid_yaml', 'high',
                f"Invalid YAML syntax: {str(e)}",
                "Fix YAML syntax errors"
            )
            
    def _analyze_json_config(self, file_path: Path) -> None:
        """Analyze JSON configuration file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        try:
            config = json.loads(content)
            self.config_files[str(file_path)] = config
            self._check_config_content(file_path, config, content)
        except json.JSONDecodeError as e:
            self._add_issue(
                str(file_path), e.lineno, 'invalid_json', 'high',
                f"Invalid JSON syntax: {str(e)}",
                "Fix JSON syntax errors"
            )
            
    def _analyze_ini_config(self, file_path: Path) -> None:
        """Analyze INI configuration file."""
        config = configparser.ConfigParser()
        
        try:
            config.read(file_path)
            config_dict = {s: dict(config.items(s)) for s in config.sections()}
            self.config_files[str(file_path)] = config_dict
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self._check_config_content(file_path, config_dict, content)
        except configparser.Error as e:
            self._add_issue(
                str(file_path), 0, 'invalid_ini', 'high',
                f"Invalid INI syntax: {str(e)}",
                "Fix INI syntax errors"
            )
            
    def _analyze_toml_config(self, file_path: Path) -> None:
        """Analyze TOML configuration file."""
        try:
            import toml
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                config = toml.loads(content)
                
            self.config_files[str(file_path)] = config
            self._check_config_content(file_path, config, content)
        except Exception as e:
            self._add_issue(
                str(file_path), 0, 'invalid_toml', 'high',
                f"Invalid TOML syntax: {str(e)}",
                "Fix TOML syntax errors or install toml package"
            )
            
    def _analyze_env_file(self, file_path: Path) -> None:
        """Analyze .env file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        config = {}
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    config[key] = value
                    
                    # Check for sensitive values
                    self._check_sensitive_value(file_path, line_num, key, value)
                else:
                    self._add_issue(
                        str(file_path), line_num, 'invalid_env_format', 'medium',
                        f"Invalid environment variable format: {line}",
                        "Use KEY=value format"
                    )
                    
        self.config_files[str(file_path)] = config
        
    def _analyze_python_config(self, file_path: Path) -> None:
        """Analyze Python configuration file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        try:
            tree = ast.parse(content, filename=str(file_path))
            
            # Extract configuration variables
            config = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            # Configuration variables are usually uppercase
                            key = target.id
                            try:
                                value = ast.literal_eval(node.value)
                                config[key] = value
                                
                                # Check line number for sensitive values
                                self._check_sensitive_value(
                                    file_path, node.lineno, key, str(value)
                                )
                            except:
                                # Complex expressions
                                pass
                                
            self.config_files[str(file_path)] = config
            
            # Check for environment variable usage
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if self._is_env_var_access(node):
                        self.env_vars_used.add(self._get_env_var_name(node))
                        
        except SyntaxError as e:
            self._add_issue(
                str(file_path), e.lineno or 0, 'syntax_error', 'high',
                f"Python syntax error: {str(e)}",
                "Fix Python syntax errors"
            )
            
    def _check_config_content(self, file_path: Path, config: Dict[str, Any], 
                             content: str) -> None:
        """Check configuration content for issues."""
        # Flatten nested configuration
        flat_config = self._flatten_dict(config)
        
        # Update metrics
        self.metrics.total_settings += len(flat_config)
        
        # Check each configuration value
        for key, value in flat_config.items():
            # Check for hardcoded secrets
            if isinstance(value, str):
                self._check_sensitive_value(file_path, 0, key, value)
                
            # Check for environment variable references
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                self.metrics.environment_vars += 1
                
            # Check for documentation (comments near the key)
            if self._has_documentation(content, key):
                self.metrics.documented_settings += 1
                
    def _check_sensitive_value(self, file_path: Path, line_number: int, 
                               key: str, value: str) -> None:
        """Check for hardcoded sensitive values."""
        # Check if key name suggests sensitive data
        key_lower = key.lower()
        is_sensitive_key = any(
            re.search(pattern, key_lower) 
            for pattern in self.sensitive_patterns
        )
        
        if is_sensitive_key:
            # Check if value looks like a real secret (not placeholder)
            if value and not self._is_placeholder(value):
                # Check if it's an environment variable reference
                if not (value.startswith('${') or value.startswith('$')):
                    self.metrics.hardcoded_secrets += 1
                    self._add_issue(
                        str(file_path), line_number, 'hardcoded_secret', 'critical',
                        f"Potential hardcoded secret in '{key}'",
                        "Use environment variables or secure key management",
                        key
                    )
                    
    def _is_placeholder(self, value: str) -> bool:
        """Check if value is a placeholder."""
        placeholders = [
            'xxx', 'todo', 'changeme', 'placeholder',
            'your_', 'my_', 'example', 'test', 'demo',
            '<', '>'
        ]
        value_lower = value.lower()
        return any(ph in value_lower for ph in placeholders)
        
    def _analyze_env_var_usage(self) -> None:
        """Analyze environment variable usage in code."""
        # Find all Python files
        python_files = list(self.project_root.rglob('*.py'))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = ast.parse(content, filename=str(file_path))
                
                # Find os.environ usage
                for node in ast.walk(tree):
                    if isinstance(node, ast.Subscript):
                        if self._is_env_var_access(node):
                            var_name = self._get_env_var_name(node)
                            if var_name:
                                self.env_vars_used.add(var_name)
                                
                    elif isinstance(node, ast.Call):
                        # Check for os.getenv() calls
                        if self._is_getenv_call(node):
                            var_name = self._get_env_var_from_call(node)
                            if var_name:
                                self.env_vars_used.add(var_name)
                                
            except:
                pass
                
    def _analyze_config_usage(self) -> None:
        """Analyze configuration key usage in code."""
        # This is a simplified analysis
        # In practice, would need more sophisticated tracking
        python_files = list(self.project_root.rglob('*.py'))
        
        # Collect all config keys
        all_config_keys = set()
        for config in self.config_files.values():
            flat = self._flatten_dict(config)
            all_config_keys.update(flat.keys())
            
        # Simple pattern matching for config usage
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for key in all_config_keys:
                    if key in content:
                        self.config_keys_used.add(key)
                        
            except:
                pass
                
    def _check_configuration_issues(self) -> None:
        """Check for various configuration issues."""
        # Check for missing required configurations
        for category, required in self.required_keys.items():
            for key in required:
                found = False
                for config in self.config_files.values():
                    flat = self._flatten_dict(config)
                    if any(k.endswith(key) for k in flat.keys()):
                        found = True
                        break
                        
                if not found:
                    self.metrics.missing_required.append(f"{category}.{key}")
                    
        # Check for unused configurations
        all_config_keys = set()
        for config in self.config_files.values():
            flat = self._flatten_dict(config)
            all_config_keys.update(flat.keys())
            
        unused = all_config_keys - self.config_keys_used
        self.metrics.unused_settings = list(unused)
        
        # Check for undefined environment variables
        env_vars_defined = set()
        for config in self.config_files.values():
            if isinstance(config, dict):
                env_vars_defined.update(
                    k for k in config.keys() 
                    if k.isupper()
                )
                
        undefined_env_vars = self.env_vars_used - env_vars_defined
        for var in undefined_env_vars:
            self._add_issue(
                "project", 0, 'undefined_env_var', 'high',
                f"Environment variable '{var}' used but not defined",
                "Define the variable in .env or configuration files"
            )
            
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', 
                      sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
        
    def _has_documentation(self, content: str, key: str) -> bool:
        """Check if configuration key has documentation."""
        # Simple check for comments near the key
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if key in line:
                # Check previous line for comment
                if i > 0 and '#' in lines[i-1]:
                    return True
                # Check same line for comment
                if '#' in line:
                    return True
        return False
        
    def _is_env_var_access(self, node: ast.AST) -> bool:
        """Check if node is environment variable access."""
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Attribute):
                if isinstance(node.value.value, ast.Name) and node.value.value.id == 'os':
                    return node.value.attr == 'environ'
        return False
        
    def _is_getenv_call(self, node: ast.Call) -> bool:
        """Check if node is os.getenv() call."""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'os':
                return node.func.attr in ['getenv', 'get']
        return False
        
    def _get_env_var_name(self, node: ast.Subscript) -> Optional[str]:
        """Extract environment variable name from subscript."""
        if isinstance(node.slice, ast.Constant):
            return str(node.slice.value)
        return None
        
    def _get_env_var_from_call(self, node: ast.Call) -> Optional[str]:
        """Extract environment variable name from getenv call."""
        if node.args and isinstance(node.args[0], ast.Constant):
            return str(node.args[0].value)
        return None
        
    def _add_issue(self, file_path: str, line_number: int, issue_type: str,
                   severity: str, message: str, suggestion: str, 
                   config_key: Optional[str] = None) -> None:
        """Add a configuration issue."""
        self.issues.append(ConfigurationIssue(
            file_path=file_path,
            line_number=line_number,
            issue_type=issue_type,
            severity=severity,
            message=message,
            suggestion=suggestion,
            config_key=config_key
        ))
        
    def _generate_report(self) -> Dict[str, Any]:
        """Generate configuration analysis report."""
        return {
            'summary': {
                'total_config_files': len(self.config_files),
                'total_settings': self.metrics.total_settings,
                'documented_settings': self.metrics.documented_settings,
                'documentation_percentage': (
                    self.metrics.documented_settings / self.metrics.total_settings * 100
                    if self.metrics.total_settings > 0 else 0
                ),
                'environment_vars_used': len(self.env_vars_used),
                'hardcoded_secrets': self.metrics.hardcoded_secrets,
                'missing_required': len(self.metrics.missing_required),
                'unused_settings': len(self.metrics.unused_settings),
                'total_issues': len(self.issues)
            },
            'issues': [
                {
                    'file': issue.file_path,
                    'line': issue.line_number,
                    'type': issue.issue_type,
                    'severity': issue.severity,
                    'message': issue.message,
                    'suggestion': issue.suggestion,
                    'config_key': issue.config_key
                }
                for issue in self.issues
            ],
            'missing_required_configs': self.metrics.missing_required,
            'unused_configs': self.metrics.unused_settings,
            'config_files': list(self.config_files.keys()),
            'security_summary': {
                'hardcoded_secrets': self.metrics.hardcoded_secrets,
                'critical_issues': len([i for i in self.issues if i.severity == 'critical'])
            }
        }