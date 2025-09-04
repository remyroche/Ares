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

from analyzers.static_analysis_analyzer import StaticAnalysisAnalyzer
from analyzers.ast_analysis_analyzer import ASTAnalysisAnalyzer
from core.config import CodeQualityConfig, StaticAnalysisConfig, ASTAnalysisConfig
from fixers.sequential_fixer import SequentialFixer
from pipelines.pipeline_unified_enhanced import UnifiedEnhancedPipeline


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


class TestEnhancedSequentialFixer(unittest.TestCase):
    """Test the enhanced SequentialFixer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CodeQualityConfig()
        self.fixer = SequentialFixer(self.config)
        
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
        self.assertIsInstance(self.fixer, SequentialFixer)
        self.assertEqual(self.fixer.config, self.config)

    @patch('code_quality.fixers.sequential_fixer.StaticAnalysisAnalyzer')
    @patch('code_quality.fixers.sequential_fixer.ASTAnalysisAnalyzer')
    def test_run_static_analysis(self, mock_ast_analyzer, mock_static_analyzer):
        """Test static analysis step."""
        # Mock the analyzer
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.analyze_directory.return_value = {
            "files": {
                str(self.test_file1): {
                    "summary": {"total_issues": 2, "critical_issues": 1, "security_issues": 0}
                }
            },
            "summary": {
                "tools_summary": {
                    "pylint": {"files_analyzed": 1, "issues_found": 1, "errors": 0},
                    "flake8": {"files_analyzed": 1, "issues_found": 1, "errors": 0}
                }
            }
        }
        mock_static_analyzer.return_value = mock_analyzer_instance

        result = self.fixer._run_static_analysis([str(self.test_file1)])
        
        self.assertEqual(result["status"], "success")
        self.assertIn("results", result)
        self.assertIn("summary", result["results"])

    @patch('code_quality.fixers.sequential_fixer.ASTAnalysisAnalyzer')
    def test_run_ast_analysis(self, mock_ast_analyzer):
        """Test AST analysis step."""
        # Mock the analyzer
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.analyze_directory.return_value = {
            "files": {
                str(self.test_file1): {
                    "summary": {
                        "total_issues": 1,
                        "complexity_issues": 1,
                        "refactoring_opportunities": 0,
                        "code_completion_issues": 0,
                        "ast_analysis_issues": 0
                    }
                }
            },
            "summary": {
                "tools_availability": {
                    "astroid": True,
                    "rope": False,
                    "jedi": True
                }
            }
        }
        mock_ast_analyzer.return_value = mock_analyzer_instance

        result = self.fixer._run_ast_analysis([str(self.test_file1)])
        
        self.assertEqual(result["status"], "success")
        self.assertIn("results", result)
        self.assertIn("summary", result["results"])

    def test_generate_comprehensive_summary(self):
        """Test comprehensive summary generation with new metrics."""
        # Set up mock results
        self.fixer.results = {
            "step_results": {
                "static_analysis": {
                    "status": "success",
                    "results": {
                        "summary": {
                            "total_issues_found": 5,
                            "critical_issues": 2,
                            "security_issues": 1
                        }
                    }
                },
                "ast_analysis": {
                    "status": "success",
                    "results": {
                        "summary": {
                            "total_issues_found": 3,
                            "complexity_issues": 2,
                            "refactoring_opportunities": 1
                        }
                    }
                }
            }
        }

        summary = self.fixer._generate_comprehensive_summary()
        
        self.assertIn("metrics", summary)
        self.assertIn("static_analysis_issues", summary["metrics"])
        self.assertIn("static_analysis_critical", summary["metrics"])
        self.assertIn("static_analysis_security", summary["metrics"])
        self.assertIn("ast_analysis_issues", summary["metrics"])
        self.assertIn("ast_analysis_complexity", summary["metrics"])
        self.assertIn("ast_analysis_refactoring", summary["metrics"])


class TestEnhancedUnifiedPipeline(unittest.TestCase):
    """Test the enhanced UnifiedEnhancedPipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.pipeline = UnifiedEnhancedPipeline(self.test_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.test_dir)

    def test_pipeline_initialization(self):
        """Test that the pipeline initializes correctly."""
        self.assertIsInstance(self.pipeline, UnifiedEnhancedPipeline)
        self.assertEqual(str(self.pipeline.project_root), self.test_dir)

    @patch('code_quality.pipelines.pipeline_unified_enhanced.StaticAnalysisAnalyzer')
    def test_run_static_analysis(self, mock_static_analyzer):
        """Test static analysis method."""
        # Mock the analyzer
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.analyze_directory.return_value = {
            "summary": {"total_files_analyzed": 1, "total_issues_found": 2}
        }
        mock_static_analyzer.return_value = mock_analyzer_instance

        result = self.pipeline.run_static_analysis()
        
        self.assertIn("execution_time", result)
        self.assertIn("summary", result)

    @patch('code_quality.pipelines.pipeline_unified_enhanced.ASTAnalysisAnalyzer')
    def test_run_ast_analysis(self, mock_ast_analyzer):
        """Test AST analysis method."""
        # Mock the analyzer
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.analyze_directory.return_value = {
            "summary": {
                "total_files_analyzed": 1,
                "total_issues_found": 1,
                "complexity_issues": 1
            }
        }
        mock_ast_analyzer.return_value = mock_analyzer_instance

        result = self.pipeline.run_ast_analysis()
        
        self.assertIn("execution_time", result)
        self.assertIn("summary", result)


class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration integration for new analysis tools."""

    def test_static_analysis_config(self):
        """Test StaticAnalysisConfig initialization."""
        config = StaticAnalysisConfig()
        
        self.assertTrue(config.enabled)
        self.assertIn("pylint", config.tools)
        self.assertIn("flake8", config.tools)
        self.assertIn("mypy", config.tools)
        self.assertIn("bandit", config.tools)
        self.assertIn("max_line_length", config.pylint_config)
        self.assertIn("extend_ignore", config.flake8_config)

    def test_ast_analysis_config(self):
        """Test ASTAnalysisConfig initialization."""
        config = ASTAnalysisConfig()
        
        self.assertTrue(config.enabled)
        self.assertIn("astroid", config.tools)
        self.assertIn("rope", config.tools)
        self.assertIn("jedi", config.tools)
        self.assertIn("custom_ast", config.tools)
        self.assertIn("max_function_length", config.astroid_config)
        self.assertIn("max_cyclomatic_complexity", config.custom_ast_config)

    def test_code_quality_config_integration(self):
        """Test that CodeQualityConfig includes new analysis configs."""
        config = CodeQualityConfig()
        
        self.assertIsInstance(config.analysis.static_analysis, StaticAnalysisConfig)
        self.assertIsInstance(config.analysis.ast_analysis, ASTAnalysisConfig)
        self.assertTrue(config.analysis.static_analysis.enabled)
        self.assertTrue(config.analysis.ast_analysis.enabled)


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestStaticAnalysisAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestASTAnalysisAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestEnhancedSequentialFixer))
    test_suite.addTest(unittest.makeSuite(TestEnhancedUnifiedPipeline))
    test_suite.addTest(unittest.makeSuite(TestConfigurationIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")