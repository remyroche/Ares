"""
Configuration for Code Complexity Analysis
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ComplexityConfig:
    """Configuration class for complexity analysis"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Set up paths
        self.base_dir = Path(__file__).parent.parent
        self.output_dir = os.path.join(self.base_dir, 'reports')
        self.reports_dir = os.path.join(self.base_dir, 'reports')
        self.logs_dir = os.path.join(self.base_dir, 'logs')
        
        # Tool settings
        self.enable_pyexamine = self.config.get('tools', {}).get('pyexamine', {}).get('enabled', True)
        self.enable_radon = self.config.get('tools', {}).get('radon', {}).get('enabled', True)
        self.enable_xenon = self.config.get('tools', {}).get('xenon', {}).get('enabled', True)
        
        # Analysis settings
        self.analysis_settings = self.config.get('analysis', {})
        self.include_tests = self.analysis_settings.get('include_tests', False)
        self.include_docs = self.analysis_settings.get('include_docs', False)
        self.max_file_size_mb = self.analysis_settings.get('max_file_size_mb', 10)
        self.max_line_count = self.analysis_settings.get('max_line_count', 10000)
        
        # Thresholds
        self.thresholds = self.config.get('thresholds', {})
        self.complexity_threshold = self.thresholds.get('complexity', 0.5)
        self.cyclomatic_complexity_threshold = self.thresholds.get('cyclomatic_complexity', 10)
        self.maintainability_threshold = self.thresholds.get('maintainability', 50)
        
        # Output settings
        self.output_settings = self.config.get('output', {})
        self.generate_json = self.output_settings.get('json', True)
        self.generate_html = self.output_settings.get('html', True)
        self.generate_markdown = self.output_settings.get('markdown', True)
        self.generate_summary = self.output_settings.get('summary', True)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading config file {self.config_path}: {e}")
                print("Using default configuration")
                
        return self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'tools': {
                'pyexamine': {
                    'enabled': True,
                    'timeout': 30,
                    'options': []
                },
                'radon': {
                    'enabled': True,
                    'timeout': 30,
                    'options': ['--json']
                },
                'xenon': {
                    'enabled': True,
                    'timeout': 30,
                    'options': ['--json']
                }
            },
            'analysis': {
                'include_tests': False,
                'include_docs': False,
                'max_file_size_mb': 10,
                'max_line_count': 10000,
                'recursive': True
            },
            'thresholds': {
                'complexity': 0.5,
                'cyclomatic_complexity': 10,
                'maintainability': 50,
                'xenon_score': 5.0
            },
            'output': {
                'json': True,
                'html': True,
                'markdown': True,
                'summary': True,
                'include_details': True
            },
            'ignore_patterns': [
                '__pycache__',
                '.git',
                '.pytest_cache',
                'venv',
                'env',
                '.venv',
                '.env',
                'node_modules',
                '.tox',
                'build',
                'dist',
                '*.egg-info'
            ]
        }
        
    def save_config(self, output_path: str):
        """Save current configuration to file"""
        try:
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"Error saving config to {output_path}: {e}")
            
    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """Get configuration for a specific tool"""
        return self.config.get('tools', {}).get(tool_name, {})
        
    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is enabled"""
        return self.config.get('tools', {}).get(tool_name, {}).get('enabled', True)
        
    def get_threshold(self, threshold_name: str) -> float:
        """Get threshold value"""
        return self.thresholds.get(threshold_name, 0.0)
        
    def should_ignore_file(self, file_path: str) -> bool:
        """Check if a file should be ignored"""
        ignore_patterns = self.config.get('ignore_patterns', [])
        
        for pattern in ignore_patterns:
            if pattern in file_path:
                return True
                
        return False
        
    def should_include_file(self, file_path: str) -> bool:
        """Check if a file should be included in analysis"""
        # Check ignore patterns
        if self.should_ignore_file(file_path):
            return False
            
        # Check if it's a test file
        if not self.include_tests and ('test_' in file_path or '_test.py' in file_path):
            return False
            
        # Check if it's a documentation file
        if not self.include_docs and ('docs' in file_path or 'doc' in file_path):
            return False
            
        return True