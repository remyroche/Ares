"""
Radon Analyzer for Code Complexity Analysis
"""

import os
import subprocess
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class RadonAnalyzer:
    """Analyzer for Radon complexity metrics"""
    
    def __init__(self, config):
        """Initialize Radon analyzer"""
        self.config = config
        self.tool_name = "radon"
        
    def is_available(self) -> bool:
        """Check if Radon is available"""
        try:
            result = subprocess.run(['radon', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
            
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single file with Radon"""
        if not self.is_available():
            logger.warning("Radon is not available")
            return {}
            
        results = {}
        
        try:
            # Cyclomatic Complexity
            cc_result = self._get_cyclomatic_complexity(file_path)
            if cc_result:
                results['cyclomatic_complexity'] = cc_result
                
            # Maintainability Index
            mi_result = self._get_maintainability_index(file_path)
            if mi_result:
                results['maintainability_index'] = mi_result
                
            # Raw metrics
            raw_result = self._get_raw_metrics(file_path)
            if raw_result:
                results['raw_metrics'] = raw_result
                
        except Exception as e:
            logger.error(f"Error running Radon on {file_path}: {e}")
            
        return results
        
    def _get_cyclomatic_complexity(self, file_path: str) -> Optional[float]:
        """Get cyclomatic complexity for a file"""
        try:
            cmd = ['radon', 'cc', '--json', file_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return None
                
            data = json.loads(result.stdout)
            
            # Calculate average complexity
            complexities = []
            for item in data:
                if 'complexity' in item:
                    complexities.append(item['complexity'])
                    
            if complexities:
                return sum(complexities) / len(complexities)
                
        except Exception as e:
            logger.error(f"Error getting cyclomatic complexity for {file_path}: {e}")
            
        return None
        
    def _get_maintainability_index(self, file_path: str) -> Optional[float]:
        """Get maintainability index for a file"""
        try:
            cmd = ['radon', 'mi', '--json', file_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return None
                
            data = json.loads(result.stdout)
            
            # Get the maintainability index
            if data and 'mi' in data[0]:
                return float(data[0]['mi'])
                
        except Exception as e:
            logger.error(f"Error getting maintainability index for {file_path}: {e}")
            
        return None
        
    def _get_raw_metrics(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get raw metrics for a file"""
        try:
            cmd = ['radon', 'raw', '--json', file_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return None
                
            data = json.loads(result.stdout)
            
            if data:
                return data[0]
                
        except Exception as e:
            logger.error(f"Error getting raw metrics for {file_path}: {e}")
            
        return None
        
    def analyze_directory(self, directory_path: str) -> Dict[str, Dict[str, Any]]:
        """Analyze all Python files in a directory"""
        results = {}
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    analysis = self.analyze_file(file_path)
                    if analysis:
                        results[file_path] = analysis
                        
        return results
        
    def get_function_complexity(self, file_path: str) -> List[Dict[str, Any]]:
        """Get complexity metrics for individual functions"""
        if not self.is_available():
            return []
            
        try:
            cmd = ['radon', 'cc', '--json', '--show-complexity', file_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return []
                
            return json.loads(result.stdout)
            
        except Exception as e:
            logger.error(f"Error getting function complexity for {file_path}: {e}")
            return []
            
    def get_halstead_metrics(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get Halstead metrics for a file"""
        if not self.is_available():
            return None
            
        try:
            cmd = ['radon', 'hal', '--json', file_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return None
                
            data = json.loads(result.stdout)
            if data:
                return data[0]
                
        except Exception as e:
            logger.error(f"Error getting Halstead metrics for {file_path}: {e}")
            
        return None