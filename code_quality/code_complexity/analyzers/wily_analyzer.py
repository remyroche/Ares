"""
Wily Analyzer for Code Complexity Analysis
Historical complexity tracking and trend analysis
"""

import subprocess
import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class WilyAnalyzer:
    """Analyzer for Wily historical complexity tracking
    
    Wily provides:
    - Historical complexity tracking
    - Trend analysis over time
    - Git integration for version tracking
    - Complexity evolution visualization
    - Regression detection
    """
    
    def __init__(self, config):
        """Initialize Wily analyzer"""
        self.config = config
        self.tool_name = "wily"
        
    def is_available(self) -> bool:
        """Check if Wily is available"""
        try:
            result = subprocess.run(['wily', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
            
    def analyze_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Analyze a single file with Wily historical tracking"""
        if not self.is_available():
            logger.warning("Wily is not available")
            return None
            
        try:
            # Get current complexity metrics
            current_metrics = self._get_current_metrics(file_path)
            
            # Get historical trends
            historical_trends = self._get_historical_trends(file_path)
            
            # Get complexity evolution
            evolution = self._get_complexity_evolution(file_path)
            
            return {
                'current_metrics': current_metrics,
                'historical_trends': historical_trends,
                'evolution': evolution,
                'analysis_timestamp': self._get_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Error running Wily on {file_path}: {e}")
            return None
            
    def _get_current_metrics(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get current complexity metrics for a file"""
        try:
            cmd = ['wily', 'report', file_path, '--format', 'json']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return None
                
            data = json.loads(result.stdout)
            return data.get('current', {})
            
        except Exception as e:
            logger.error(f"Error getting current metrics for {file_path}: {e}")
            return None
            
    def _get_historical_trends(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        """Get historical complexity trends for a file"""
        try:
            cmd = ['wily', 'history', file_path, '--format', 'json']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return None
                
            data = json.loads(result.stdout)
            return data.get('history', [])
            
        except Exception as e:
            logger.error(f"Error getting historical trends for {file_path}: {e}")
            return None
            
    def _get_complexity_evolution(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get complexity evolution analysis"""
        try:
            cmd = ['wily', 'diff', file_path, '--format', 'json']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return None
                
            data = json.loads(result.stdout)
            return data
            
        except Exception as e:
            logger.error(f"Error getting complexity evolution for {file_path}: {e}")
            return None
            
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
        
    def build_history(self, directory_path: str) -> bool:
        """Build historical complexity database for a directory"""
        if not self.is_available():
            logger.warning("Wily is not available")
            return False
            
        try:
            cmd = ['wily', 'build', directory_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info(f"Successfully built Wily history for {directory_path}")
                return True
            else:
                logger.error(f"Failed to build Wily history: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error building Wily history for {directory_path}: {e}")
            return False
            
    def get_regression_analysis(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get complexity regression analysis"""
        if not self.is_available():
            return None
            
        try:
            cmd = ['wily', 'regression', file_path, '--format', 'json']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return None
                
            return json.loads(result.stdout)
            
        except Exception as e:
            logger.error(f"Error getting regression analysis for {file_path}: {e}")
            return None