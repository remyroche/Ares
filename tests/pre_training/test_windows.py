"""
Test Window Quality Assessment Utilities

Tests for window validation and quality assessment in the Analyst→Tactician pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Import validation utilities
from src.utils.ml_common.validation.windows import (
    validate_window_structure,
    calculate_window_statistics,
    detect_window_quality_issues,
    assess_windows,
    validate_window_quality
)


class TestWindowStructureValidation:
    """Test window structure validation."""
    
    def test_valid_window_structure(self):
        """Test validation of a properly structured window."""
        window = {
            'start': pd.Timestamp('2023-01-01 10:00:00'),
            'end': pd.Timestamp('2023-01-01 10:05:00'),
            'anchor': pd.Timestamp('2023-01-01 10:02:00'),
            'direction': 1
        }
        
        is_valid, errors = validate_window_structure(window)
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_window_missing_keys(self):
        """Test validation of window with missing required keys."""
        window = {
            'start': pd.Timestamp('2023-01-01 10:00:00'),
            'end': pd.Timestamp('2023-01-01 10:05:00'),
            # Missing 'anchor' and 'direction'
        }
        
        is_valid, errors = validate_window_structure(window)
        assert not is_valid
        assert len(errors) > 0
        assert any('Missing required keys' in error for error in errors)
    
    def test_invalid_window_temporal_ordering(self):
        """Test validation of window with invalid temporal ordering."""
        window = {
            'start': pd.Timestamp('2023-01-01 10:05:00'),  # After end
            'end': pd.Timestamp('2023-01-01 10:00:00'),
            'anchor': pd.Timestamp('2023-01-01 10:02:00'),
            'direction': 1
        }
        
        is_valid, errors = validate_window_structure(window)
        assert not is_valid
        assert any('Start timestamp' in error for error in errors)
    
    def test_invalid_window_anchor_outside_range(self):
        """Test validation of window with anchor outside start-end range."""
        window = {
            'start': pd.Timestamp('2023-01-01 10:00:00'),
            'end': pd.Timestamp('2023-01-01 10:05:00'),
            'anchor': pd.Timestamp('2023-01-01 10:06:00'),  # After end
            'direction': 1
        }
        
        is_valid, errors = validate_window_structure(window)
        assert not is_valid
        assert any('Anchor timestamp' in error for error in errors)
    
    def test_invalid_window_direction(self):
        """Test validation of window with invalid direction."""
        window = {
            'start': pd.Timestamp('2023-01-01 10:00:00'),
            'end': pd.Timestamp('2023-01-01 10:05:00'),
            'anchor': pd.Timestamp('2023-01-01 10:02:00'),
            'direction': 0  # Invalid direction (should be -1 or 1)
        }
        
        is_valid, errors = validate_window_structure(window)
        assert not is_valid
        assert any('Invalid direction' in error for error in errors)
    
    def test_invalid_window_too_short(self):
        """Test validation of window that's too short."""
        window = {
            'start': pd.Timestamp('2023-01-01 10:00:00'),
            'end': pd.Timestamp('2023-01-01 10:00:30'),  # 30 seconds
            'anchor': pd.Timestamp('2023-01-01 10:00:15'),
            'direction': 1
        }
        
        is_valid, errors = validate_window_structure(window)
        assert not is_valid
        assert any('Window too short' in error for error in errors)


class TestWindowStatistics:
    """Test window statistics calculation."""
    
    def test_calculate_window_statistics_empty(self):
        """Test statistics calculation with empty window list."""
        windows = []
        stats = calculate_window_statistics(windows)
        
        assert stats['total_windows'] == 0
        assert stats['valid_windows'] == 0
        assert stats['coverage_ratio'] == 0.0
    
    def test_calculate_window_statistics_valid_windows(self):
        """Test statistics calculation with valid windows."""
        windows = [
            {
                'start': pd.Timestamp('2023-01-01 10:00:00'),
                'end': pd.Timestamp('2023-01-01 10:05:00'),
                'anchor': pd.Timestamp('2023-01-01 10:02:00'),
                'direction': 1
            },
            {
                'start': pd.Timestamp('2023-01-01 11:00:00'),
                'end': pd.Timestamp('2023-01-01 11:05:00'),
                'anchor': pd.Timestamp('2023-01-01 11:02:00'),
                'direction': -1
            }
        ]
        
        stats = calculate_window_statistics(windows)
        
        assert stats['total_windows'] == 2
        assert stats['valid_windows'] == 2
        assert stats['invalid_windows'] == 0
        assert stats['directions']['long_count'] == 1
        assert stats['directions']['short_count'] == 1
    
    def test_calculate_window_statistics_with_overlaps(self):
        """Test statistics calculation with overlapping windows."""
        windows = [
            {
                'start': pd.Timestamp('2023-01-01 10:00:00'),
                'end': pd.Timestamp('2023-01-01 10:05:00'),
                'anchor': pd.Timestamp('2023-01-01 10:02:00'),
                'direction': 1
            },
            {
                'start': pd.Timestamp('2023-01-01 10:03:00'),  # Overlaps with first
                'end': pd.Timestamp('2023-01-01 10:08:00'),
                'anchor': pd.Timestamp('2023-01-01 10:05:00'),
                'direction': 1
            }
        ]
        
        stats = calculate_window_statistics(windows)
        
        assert stats['overlap_count'] == 1
        assert stats['valid_windows'] == 2
    
    def test_calculate_window_statistics_with_data_index(self):
        """Test statistics calculation with data index for coverage."""
        windows = [
            {
                'start': pd.Timestamp('2023-01-01 10:00:00'),
                'end': pd.Timestamp('2023-01-01 10:05:00'),
                'anchor': pd.Timestamp('2023-01-01 10:02:00'),
                'direction': 1
            }
        ]
        
        # Create data index with 10 timestamps
        data_index = pd.date_range(
            start='2023-01-01 09:55:00',
            end='2023-01-01 10:10:00',
            freq='1min'
        )
        
        stats = calculate_window_statistics(windows, data_index)
        
        assert stats['coverage_ratio'] > 0.0
        assert stats['coverage_ratio'] <= 1.0


class TestWindowQualityIssues:
    """Test window quality issue detection."""
    
    def test_detect_no_windows(self):
        """Test detection of no windows issue."""
        windows = []
        issues = detect_window_quality_issues(windows)
        
        assert len(issues) == 1
        assert issues[0]['type'] == 'no_windows'
        assert issues[0]['severity'] == 'high'
    
    def test_detect_invalid_window_structure(self):
        """Test detection of invalid window structure issues."""
        windows = [
            {
                'start': pd.Timestamp('2023-01-01 10:00:00'),
                'end': pd.Timestamp('2023-01-01 10:05:00'),
                'anchor': pd.Timestamp('2023-01-01 10:02:00'),
                'direction': 1
            },
            {
                'start': pd.Timestamp('2023-01-01 11:00:00'),
                # Missing required keys
                'anchor': pd.Timestamp('2023-01-01 11:02:00'),
                'direction': 1
            }
        ]
        
        issues = detect_window_quality_issues(windows)
        
        invalid_structure_issues = [issue for issue in issues if issue['type'] == 'invalid_window_structure']
        assert len(invalid_structure_issues) == 1
        assert invalid_structure_issues[0]['severity'] == 'high'
    
    def test_detect_window_overlaps(self):
        """Test detection of window overlap issues."""
        windows = [
            {
                'start': pd.Timestamp('2023-01-01 10:00:00'),
                'end': pd.Timestamp('2023-01-01 10:05:00'),
                'anchor': pd.Timestamp('2023-01-01 10:02:00'),
                'direction': 1
            },
            {
                'start': pd.Timestamp('2023-01-01 10:03:00'),  # Overlaps
                'end': pd.Timestamp('2023-01-01 10:08:00'),
                'anchor': pd.Timestamp('2023-01-01 10:05:00'),
                'direction': 1
            }
        ]
        
        issues = detect_window_quality_issues(windows)
        
        overlap_issues = [issue for issue in issues if issue['type'] == 'window_overlap']
        assert len(overlap_issues) == 1
        assert overlap_issues[0]['severity'] == 'medium'
    
    def test_detect_duplicate_anchors(self):
        """Test detection of duplicate anchor issues."""
        windows = [
            {
                'start': pd.Timestamp('2023-01-01 10:00:00'),
                'end': pd.Timestamp('2023-01-01 10:05:00'),
                'anchor': pd.Timestamp('2023-01-01 10:02:00'),
                'direction': 1
            },
            {
                'start': pd.Timestamp('2023-01-01 11:00:00'),
                'end': pd.Timestamp('2023-01-01 11:05:00'),
                'anchor': pd.Timestamp('2023-01-01 10:02:00'),  # Duplicate anchor
                'direction': 1
            }
        ]
        
        issues = detect_window_quality_issues(windows)
        
        duplicate_issues = [issue for issue in issues if issue['type'] == 'duplicate_anchors']
        assert len(duplicate_issues) == 1
        assert duplicate_issues[0]['severity'] == 'medium'


class TestWindowAssessment:
    """Test comprehensive window assessment."""
    
    def test_assess_windows_valid(self):
        """Test assessment of valid windows."""
        windows = [
            {
                'start': pd.Timestamp('2023-01-01 10:00:00'),
                'end': pd.Timestamp('2023-01-01 10:05:00'),
                'anchor': pd.Timestamp('2023-01-01 10:02:00'),
                'direction': 1
            },
            {
                'start': pd.Timestamp('2023-01-01 11:00:00'),
                'end': pd.Timestamp('2023-01-01 11:05:00'),
                'anchor': pd.Timestamp('2023-01-01 11:02:00'),
                'direction': -1
            }
        ]
        
        result = assess_windows(windows)
        
        assert result.is_valid
        assert result.total_windows == 2
        assert result.valid_windows == 2
        assert len(result.quality_issues) == 0
    
    def test_assess_windows_invalid(self):
        """Test assessment of invalid windows."""
        windows = [
            {
                'start': pd.Timestamp('2023-01-01 10:00:00'),
                'end': pd.Timestamp('2023-01-01 10:05:00'),
                'anchor': pd.Timestamp('2023-01-01 10:02:00'),
                'direction': 1
            },
            {
                # Invalid window - missing required keys
                'start': pd.Timestamp('2023-01-01 11:00:00'),
                'anchor': pd.Timestamp('2023-01-01 11:02:00'),
                'direction': 1
            }
        ]
        
        result = assess_windows(windows)
        
        assert not result.is_valid
        assert result.total_windows == 2
        assert result.valid_windows == 1
        assert len(result.quality_issues) > 0


class TestValidateWindowQuality:
    """Test window quality validation integration."""
    
    def test_validate_window_quality_success(self):
        """Test successful window quality validation."""
        windows = [
            {
                'start': pd.Timestamp('2023-01-01 10:00:00'),
                'end': pd.Timestamp('2023-01-01 10:05:00'),
                'anchor': pd.Timestamp('2023-01-01 10:02:00'),
                'direction': 1
            }
        ]
        
        artifacts = {
            'opportunity_windows': windows,
            'data': pd.DataFrame(index=pd.date_range('2023-01-01 09:55:00', '2023-01-01 10:10:00', freq='1min'))
        }
        
        result = validate_window_quality(artifacts)
        
        assert result['success']
        assert 'windows' in result['results']
    
    def test_validate_window_quality_no_windows(self):
        """Test window quality validation with no windows."""
        artifacts = {
            'data': pd.DataFrame(index=pd.date_range('2023-01-01 09:55:00', '2023-01-01 10:10:00', freq='1min'))
        }
        
        result = validate_window_quality(artifacts)
        
        assert not result['success']
        assert 'no_windows' in str(result['results']['windows'].quality_issues[0]['type'])
    
    def test_validate_window_quality_config_requirements(self):
        """Test window quality validation with configuration requirements."""
        windows = [
            {
                'start': pd.Timestamp('2023-01-01 10:00:00'),
                'end': pd.Timestamp('2023-01-01 10:05:00'),
                'anchor': pd.Timestamp('2023-01-01 10:02:00'),
                'direction': 1
            }
        ]
        
        artifacts = {
            'opportunity_windows': windows,
            'data': pd.DataFrame(index=pd.date_range('2023-01-01 09:55:00', '2023-01-01 10:10:00', freq='1min'))
        }
        
        config = {
            'require_min_windows': 2,  # Require at least 2 windows
            'strict_mode': True
        }
        
        result = validate_window_quality(artifacts, config)
        
        assert not result['success']  # Should fail because only 1 window provided


# Integration test for overlapping windows
class TestOverlappingWindows:
    """Test handling of overlapping windows."""
    
    def test_overlapping_windows_detection(self):
        """Test detection and handling of overlapping windows."""
        windows = [
            {
                'start': pd.Timestamp('2023-01-01 10:00:00'),
                'end': pd.Timestamp('2023-01-01 10:05:00'),
                'anchor': pd.Timestamp('2023-01-01 10:02:00'),
                'direction': 1
            },
            {
                'start': pd.Timestamp('2023-01-01 10:03:00'),  # Overlaps with first
                'end': pd.Timestamp('2023-01-01 10:08:00'),
                'anchor': pd.Timestamp('2023-01-01 10:05:00'),
                'direction': 1
            },
            {
                'start': pd.Timestamp('2023-01-01 10:10:00'),  # No overlap
                'end': pd.Timestamp('2023-01-01 10:15:00'),
                'anchor': pd.Timestamp('2023-01-01 10:12:00'),
                'direction': -1
            }
        ]
        
        issues = detect_window_quality_issues(windows)
        
        overlap_issues = [issue for issue in issues if issue['type'] == 'window_overlap']
        assert len(overlap_issues) == 1
        assert overlap_issues[0]['description'] == '1 window overlaps detected'
    
    def test_one_bar_windows(self):
        """Test handling of very short (1-bar) windows."""
        windows = [
            {
                'start': pd.Timestamp('2023-01-01 10:00:00'),
                'end': pd.Timestamp('2023-01-01 10:00:30'),  # 30 seconds
                'anchor': pd.Timestamp('2023-01-01 10:00:15'),
                'direction': 1
            }
        ]
        
        issues = detect_window_quality_issues(windows)
        
        short_window_issues = [issue for issue in issues if issue['type'] == 'short_windows']
        assert len(short_window_issues) == 1
        assert short_window_issues[0]['severity'] == 'low'


if __name__ == "__main__":
    pytest.main([__file__])
