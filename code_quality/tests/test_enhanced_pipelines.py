from src.utils.tprint import tprint

"""
Comprehensive test suite for enhanced pipelines with static analysis and AST analysis.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from code_quality.analyzers.static_analysis_analyzer import StaticAnalysisAnalyzer
from code_quality.analyzers.ast_analysis_analyzer import ASTAnalysisAnalyzer
from code_quality.core.config import CodeQualityConfig, AnalysisConfig
from code_quality.fixers.auto_fixer import AutoFixer
from code_quality.pipelines.auto_fixer_pipeline import AutoFixerPipeline
from code_quality.analysis_functions import run_all_analyses


class TestStaticAnalysisAnalyzer(unittest.TestCase):
    """Test the StaticAnalysisAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CodeQualityConfig()
        self.analyzer = StaticAnalysisAnalyzer(self.config)
        
        # Create a temporary test file
        self.test_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        self.test_file.write("""
import os
import sys

def test_function(x, y):
    # This function has some issues for testing
    unused_var = "not used"
    result = x + y
    return result

class TestClass:
    def __init__(self):
        self.value = 0
    
    def method_with_issues(self):
        # Long line that exceeds the limit
        very_long_variable_name_that_should_trigger_line_length_warning = "this is a very long string that should trigger a line length warning"
        return very_long_variable_name_that_should_trigger_line_length_warning
""")
        self.test_file.close()

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.test_file.name)

    def test_analyzer_initialization(self):
        """Test that the analyzer initializes correctly."""
        self.assertIsInstance(self.analyzer, StaticAnalysisAnalyzer)
        self.assertEqual(self.analyzer.config, self.config)
        self.assertIn("pylint", self.analyzer.tools)
        self.assertIn("flake8", self.analyzer.tools)
        self.assertIn("mypy", self.analyzer.tools)
        self.assertIn("bandit", self.analyzer.tools)

    @patch('subprocess.run')
    def test_pylint_analysis(self, mock_run):
        """Test Pylint analysis functionality."""
        # Mock successful Pylint run
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps([
            {
                "line": 5,
                "column": 4,
                "message": "Unused variable 'unused_var'",
                "type": "warning",
                "message-id": "W0612",
                "symbol": "unused-variable"
            }
        ])
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = self.analyzer._run_pylint(self.test_file.name)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("issues", result)
        self.assertGreater(len(result["issues"]), 0)

    @patch('subprocess.run')
    def test_flake8_analysis(self, mock_run):
        """Test Flake8 analysis functionality."""
        # Mock successful Flake8 run
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = f"{self.test_file.name}:8:1: E501 line too long (120 > 88 characters)"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = self.analyzer._run_flake8(self.test_file.name)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("issues", result)

    @patch('subprocess.run')
    def test_mypy_analysis(self, mock_run):
        """Test MyPy analysis functionality."""
        # Mock successful MyPy run
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = f"{self.test_file.name}:5: error: Function is missing a return type annotation"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = self.analyzer._run_mypy(self.test_file.name)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("issues", result)

    @patch('subprocess.run')
    def test_bandit_analysis(self, mock_run):
        """Test Bandit analysis functionality."""
        # Mock successful Bandit run
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "results": [
                {
                    "line_number": 3,
                    "issue_text": "Use of hardcoded password strings",
                    "issue_severity": "MEDIUM",
                    "issue_confidence": "MEDIUM",
                    "test_id": "B105"
                }
            ]
        })
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = self.analyzer._run_bandit(self.test_file.name)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("issues", result)

    def test_analyze_file(self):
        """Test file analysis with mocked tools."""
        with patch.object(self.analyzer, '_run_pylint') as mock_pylint, \
             patch.object(self.analyzer, '_run_flake8') as mock_flake8, \
             patch.object(self.analyzer, '_run_mypy') as mock_mypy, \
             patch.object(self.analyzer, '_run_bandit') as mock_bandit:
            
            # Mock all tools to return success
            mock_pylint.return_value = {"status": "success", "issues": []}
            mock_flake8.return_value = {"status": "success", "issues": []}
            mock_mypy.return_value = {"status": "success", "issues": []}
            mock_bandit.return_value = {"status": "success", "issues": []}

            result = self.analyzer.analyze_file(self.test_file.name)
            
            self.assertIn("file", result)
            self.assertIn("tools", result)
            self.assertIn("summary", result)
            self.assertEqual(result["file"], self.test_file.name)
            self.assertIn("pylint", result["tools"])
            self.assertIn("flake8", result["tools"])
            self.assertIn("mypy", result["tools"])
            self.assertIn("bandit", result["tools"])


class TestASTAnalysisAnalyzer(unittest.TestCase):
    """Test the ASTAnalysisAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CodeQualityConfig()
        self.analyzer = ASTAnalysisAnalyzer(self.config)
        
        # Create a temporary test file
        self.test_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        self.test_file.write("""
import os
import sys

def complex_function(x, y, z, a, b, c, d):
    # This function has high complexity and many parameters
    if x > 0:
        if y > 0:
            if z > 0:
                if a > 0:
                    if b > 0:
                        if c > 0:
                            if d > 0:
                                return x + y + z + a + b + c + d
                            else:
                                return x + y + z + a + b + c
                        else:
                            return x + y + z + a + b
                    else:
                        return x + y + z + a
                else:
                    return x + y + z
            else:
                return x + y
        else:
            return x
    else:
        return 0

class TestClass:
    def __init__(self):
        self.value = 0
    
    def unused_method(self):
        # This method is never called
        return "unused"
""")
        self.test_file.close()

    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.test_file.name)

    def test_analyzer_initialization(self):
        """Test that the analyzer initializes correctly."""
        self.assertIsInstance(self.analyzer, ASTAnalysisAnalyzer)
        self.assertEqual(self.analyzer.config, self.config)

    def test_custom_ast_analysis(self):
        """Test custom AST analysis functionality."""
        result = self.analyzer._run_custom_ast_analysis(self.test_file.name)
        
        self.assertEqual(result["status"], "success")
        self.assertIn("complexity_issues", result)
        self.assertIn("ast_info", result)
        
        # Should find high complexity function
        complexity_issues = result["complexity_issues"]
        self.assertGreater(len(complexity_issues), 0)
        
        # Should find function with too many parameters
        param_issues = [issue for issue in complexity_issues if issue["code"] == "too_many_parameters"]
        self.assertGreater(len(param_issues), 0)

    def test_analyze_file(self):
        """Test file analysis."""
        result = self.analyzer.analyze_file(self.test_file.name)
        
        self.assertIn("file", result)
        self.assertIn("tools", result)
        self.assertIn("summary", result)
        self.assertEqual(result["file"], self.test_file.name)
        self.assertIn("custom_ast", result["tools"])


class TestEnhancedAutoFixer(unittest.TestCase):
    """Test the enhanced AutoFixer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CodeQualityConfig()
        self.fixer = AutoFixer(self.config)
        
        # Create a temporary test directory with Python files
        self.test_dir = tempfile.mkdtemp()
        self.test_file1 = Path(self.test_dir) / "test1.py"
        self.test_file2 = Path(self.test_dir) / "test2.py"
        
        self.test_file1.write_text("""
def simple_function(x):
    return x * 2

class SimpleClass:
    def method(self):
        return "test"
""")
        
        self.test_file2.write_text("""
import os

def another_function(y):
    unused_var = "not used"
    return y + 1
""")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir)

    def test_fixer_initialization(self):
        """Test that the fixer initializes correctly."""
        self.assertIsInstance(self.fixer, AutoFixer)
        self.assertEqual(self.fixer.config, self.config)

    def test_fix_all_method(self):
        """Test fix_all method."""
        # Test that fix_all method exists and can be called
        result = self.fixer.fix_all(str(self.test_dir))
        
        self.assertIsInstance(result, dict)
        self.assertIn("files_processed", result)
        self.assertIn("total_fixes", result)

    def test_fix_file_method(self):
        """Test fix_file method."""
        # Test that fix_file method exists and can be called
        result = self.fixer.fix_file(str(self.test_file1))
        
        self.assertIsInstance(result, dict)
        self.assertIn("file", result)
        self.assertIn("fixes_applied", result)

    def test_get_fix_summary(self):
        """Test fix summary generation."""
        # Test that get_fix_summary method exists and can be called
        summary = self.fixer.get_fix_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn("total_files_processed", summary)
        self.assertIn("total_fixes_applied", summary)
        self.assertIn("fixes_by_type", summary)


class TestEnhancedAutoFixerPipeline(unittest.TestCase):
    """Test the enhanced AutoFixerPipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        from code_quality.pipelines.base_pipeline import PipelineConfig
        config = PipelineConfig()
        self.pipeline = AutoFixerPipeline(config)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir)

    def test_pipeline_initialization(self):
        """Test that the pipeline initializes correctly."""
        self.assertIsInstance(self.pipeline, AutoFixerPipeline)
        self.assertEqual(self.pipeline.config, self.pipeline.config)

    def test_pipeline_stages(self):
        """Test pipeline stages method."""
        # Test that get_stages method exists and returns expected stages
        stages = self.pipeline.get_stages()
        
        self.assertIsInstance(stages, list)
        self.assertGreater(len(stages), 0)
        
        # Check that expected stages are present
        from code_quality.pipelines.base_pipeline import PipelineStage
        expected_stages = [
            PipelineStage.INITIALIZATION,
            PipelineStage.PREPARATION,
            PipelineStage.ANALYSIS,
            PipelineStage.PROCESSING,
            PipelineStage.AGGREGATION,
            PipelineStage.REPORTING,
            PipelineStage.CLEANUP
        ]
        
        for expected_stage in expected_stages:
            self.assertIn(expected_stage, stages)

    def test_pipeline_execution(self):
        """Test pipeline execution method."""
        # Test that execute_stage method exists and can be called
        from code_quality.pipelines.base_pipeline import PipelineStage
        import asyncio
        
        async def test_execution():
            context = {}
            result = await self.pipeline.execute_stage(PipelineStage.INITIALIZATION, context)
            return result
        
        result = asyncio.run(test_execution())
        
        self.assertIsNotNone(result)
        self.assertIn("stage", result)
        self.assertIn("status", result)


class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration integration for new analysis tools."""

    def test_analysis_config(self):
        """Test AnalysisConfig initialization."""
        config = AnalysisConfig()
        
        self.assertTrue(config.enable_dead_code_analysis)
        self.assertTrue(config.enable_dependency_analysis)
        self.assertTrue(config.enable_call_graph_analysis)
        self.assertTrue(config.enable_complexity_analysis)
        self.assertEqual(config.complexity_threshold, 10)
        self.assertEqual(config.confidence_threshold, 0.8)

    def test_code_quality_config_integration(self):
        """Test that CodeQualityConfig includes analysis configs."""
        config = CodeQualityConfig()
        
        self.assertIsInstance(config.analysis_config, AnalysisConfig)
        self.assertTrue(config.analysis_config.enable_dead_code_analysis)
        self.assertTrue(config.analysis_config.enable_dependency_analysis)


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestStaticAnalysisAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestASTAnalysisAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestEnhancedAutoFixer))
    test_suite.addTest(unittest.makeSuite(TestEnhancedAutoFixerPipeline))
    test_suite.addTest(unittest.makeSuite(TestConfigurationIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    tprint(f"\n{'='*50}")
    tprint("TEST SUMMARY")
    tprint(f"{'='*50}")
    tprint(f"Tests run: {result.testsRun}")
    tprint(f"Failures: {len(result.failures)}")
    tprint(f"Errors: {len(result.errors)}")
    tprint(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        tprint("\nFAILURES:")
        for test, traceback in result.failures:
            tprint(f"- {test}: {traceback}")
    
    if result.errors:
        tprint("\nERRORS:")
        for test, traceback in result.errors:
            tprint(f"- {test}: {traceback}")