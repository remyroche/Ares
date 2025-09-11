#!/usr/bin/env python3
"""
Simple Base Class for Code Quality Pipelines

This provides minimal common functionality for code quality analysis pipelines
without the complexity of the full BasePipeline class.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Union


class SimplePipelineConfig:
    """Simple configuration class for pipelines."""
    
    def __init__(self, project_root: Union[str, Path], **kwargs):
        self.project_root = Path(project_root)
        self.output_dir = kwargs.get('output_dir', self.project_root / "code_quality" / "reports")
        self.parallel_execution = kwargs.get('parallel_execution', True)
        self.max_workers = kwargs.get('max_workers', 4)
        self.timeout_per_tool = kwargs.get('timeout_per_tool', 300)
        self.retry_attempts = kwargs.get('retry_attempts', 3)
        self.log_level = kwargs.get('log_level', 'INFO')
        self.dry_run = kwargs.get('dry_run', False)
        self.verbose = kwargs.get('verbose', False)
        self.cache_enabled = kwargs.get('cache_enabled', True)
        self.cache_dir = kwargs.get('cache_dir', None)
        
        # Ensure output directory exists
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def __dict__(self):
        """Convert to dictionary for compatibility."""
        return {
            'project_root': str(self.project_root),
            'output_dir': str(self.output_dir),
            'parallel_execution': self.parallel_execution,
            'max_workers': self.max_workers,
            'timeout_per_tool': self.timeout_per_tool,
            'retry_attempts': self.retry_attempts,
            'log_level': self.log_level,
            'dry_run': self.dry_run,
            'verbose': self.verbose,
            'cache_enabled': self.cache_enabled,
            'cache_dir': str(self.cache_dir) if self.cache_dir else None,
        }


class SimplePipeline:
    """Simple base class for code quality pipelines."""
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None, 
                 config: Optional[SimplePipelineConfig] = None,
                 pipeline_name: str = "simple_pipeline"):
        """Initialize the simple pipeline."""
        self.pipeline_name = pipeline_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup configuration
        if config is None:
            if project_root is None:
                project_root = Path.cwd()
            config = SimplePipelineConfig(project_root)
        
        self.config = config
        self.project_root = self.config.project_root
        self.reports_dir = self.config.output_dir
        
        # Setup logging
        self.logger = logging.getLogger(f"code_quality.{pipeline_name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        self.logger.info(f"Initialized {pipeline_name} for project: {self.project_root}")
    
    def save_report(self, data: Dict[str, Any], filename: str) -> Path:
        """Save a report to the reports directory."""
        report_path = self.reports_dir / f"{filename}_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        self.logger.info(f"Report saved to: {report_path}")
        return report_path
    
    def print_summary(self, data: Dict[str, Any], title: str = "Analysis Summary"):
        """Print a summary of the analysis results."""
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    print(f"{key}: {len(value)} items")
                else:
                    print(f"{key}: {value}")
        else:
            print(f"Results: {data}")
        
        print(f"{'='*60}")