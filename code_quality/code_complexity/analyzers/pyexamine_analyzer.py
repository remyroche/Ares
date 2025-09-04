"""
PyExamine Analyzer for Code Complexity Analysis
"""

import os
import subprocess
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class PyExamineAnalyzer:
    """Analyzer for PyExamine complexity metrics"""
    
    def __init__(self, config):
        """Initialize PyExamine analyzer"""
        self.config = config
        self.tool_name = "pyexamine"
        
    def is_available(self) -> bool:
        """Check if PyExamine is available"""
        try:
            result = subprocess.run(['pyexamine', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
            
    def analyze_file(self, file_path: str) -> Optional[float]:
        """Analyze a single file with PyExamine"""
        if not self.is_available():
            logger.warning("PyExamine is not available")
            return None
            
        try:
            # Run PyExamine analysis
            cmd = ['pyexamine', '--json', file_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"PyExamine failed for {file_path}: {result.stderr}")
                return None
                
            # Parse JSON output
            data = json.loads(result.stdout)
            
            # Extract complexity score (adjust based on PyExamine output format)
            if 'complexity' in data:
                return float(data['complexity'])
            elif 'score' in data:
                return float(data['score'])
            else:
                # Try to extract from nested structure
                for key, value in data.items():
                    if isinstance(value, dict) and 'complexity' in value:
                        return float(value['complexity'])
                        
            logger.warning(f"Could not extract complexity score from PyExamine output for {file_path}")
            return None
            
        except subprocess.TimeoutExpired:
            logger.error(f"PyExamine analysis timed out for {file_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse PyExamine JSON output for {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error running PyExamine on {file_path}: {e}")
            return None
            
    def analyze_directory(self, directory_path: str) -> Dict[str, float]:
        """Analyze all Python files in a directory"""
        results = {}
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    score = self.analyze_file(file_path)
                    if score is not None:
                        results[file_path] = score
                        
        return results
        
    def get_detailed_analysis(self, file_path: str) -> Dict[str, Any]:
        """Get detailed analysis results from PyExamine"""
        if not self.is_available():
            return {}
            
        try:
            cmd = ['pyexamine', '--json', '--detailed', file_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"PyExamine detailed analysis failed for {file_path}: {result.stderr}")
                return {}
                
            return json.loads(result.stdout)
            
        except Exception as e:
            logger.error(f"Error getting detailed PyExamine analysis for {file_path}: {e}")
            return {}