"""
Mock Dependencies for Testing

This module provides mock implementations of external dependencies
to allow the pipeline to run without requiring pandas, numpy, scikit-learn, etc.
"""

import sys
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Mock pandas
class MockDataFrame:
    """Mock pandas DataFrame for testing."""
    
    def __init__(self, data=None, columns=None, index=None):
        self.data = data or {}
        self.columns = columns or []
        self.index = index or list(range(len(data) if data else 0))
        self._shape = (len(self.index), len(self.columns))
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return MockSeries(self.data.get(key, []))
        elif isinstance(key, slice):
            return MockDataFrame(
                data={col: self.data.get(col, [])[key] for col in self.columns},
                columns=self.columns,
                index=self.index[key]
            )
        return self.data.get(key, [])
    
    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.data[key] = value
            if key not in self.columns:
                self.columns.append(key)
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def iloc(self, indexer):
        if isinstance(indexer, slice):
            return MockDataFrame(
                data={col: self.data.get(col, [])[indexer] for col in self.columns},
                columns=self.columns,
                index=self.index[indexer]
            )
        return self.data.get(self.columns[indexer], [])
    
    @property
    def shape(self):
        return self._shape
    
    def head(self, n=5):
        return MockDataFrame(
            data={col: self.data.get(col, [])[:n] for col in self.columns},
            columns=self.columns,
            index=self.index[:n]
        )
    
    def tail(self, n=5):
        return MockDataFrame(
            data={col: self.data.get(col, [])[-n:] for col in self.columns},
            columns=self.columns,
            index=self.index[-n:]
        )
    
    def dropna(self):
        return self  # Mock implementation
    
    def fillna(self, value):
        return self  # Mock implementation
    
    def to_dict(self, orient='dict'):
        return self.data
    
    def copy(self):
        return MockDataFrame(
            data=self.data.copy(),
            columns=self.columns.copy(),
            index=self.index.copy()
        )

class MockSeries:
    """Mock pandas Series for testing."""
    
    def __init__(self, data=None, index=None, name=None):
        self.data = data or []
        self.index = index or list(range(len(data) if data else 0))
        self.name = name
        self._shape = (len(self.data),)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            return MockSeries(self.data[key], self.index[key])
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value
    
    @property
    def shape(self):
        return self._shape
    
    def head(self, n=5):
        return MockSeries(self.data[:n], self.index[:n])
    
    def tail(self, n=5):
        return MockSeries(self.data[-n:], self.index[-n:])
    
    def dropna(self):
        return self  # Mock implementation
    
    def fillna(self, value):
        return self  # Mock implementation
    
    def to_dict(self):
        return dict(zip(self.index, self.data))
    
    def copy(self):
        return MockSeries(self.data.copy(), self.index.copy(), self.name)

# Mock numpy
class MockNumpy:
    """Mock numpy for testing."""
    
    def __init__(self):
        self.random = MockRandom()
    
    def array(self, data):
        return data
    
    def zeros(self, shape):
        if isinstance(shape, int):
            return [0.0] * shape
        elif len(shape) == 2:
            return [[0.0] * shape[1] for _ in range(shape[0])]
        return [0.0] * shape[0]
    
    def ones(self, shape):
        if isinstance(shape, int):
            return [1.0] * shape
        elif len(shape) == 2:
            return [[1.0] * shape[1] for _ in range(shape[0])]
        return [1.0] * shape[0]
    
    def random(self):
        return self.random
    
    def mean(self, data):
        if not data:
            return 0.0
        return sum(data) / len(data)
    
    def std(self, data):
        if not data:
            return 0.0
        mean_val = self.mean(data)
        variance = sum((x - mean_val) ** 2 for x in data) / len(data)
        return variance ** 0.5
    
    def max(self, data):
        return max(data) if data else 0.0
    
    def min(self, data):
        return min(data) if data else 0.0
    
    def argmax(self, data):
        return data.index(max(data)) if data else 0
    
    def argmin(self, data):
        return data.index(min(data)) if data else 0
    
    def linspace(self, start, stop, num):
        step = (stop - start) / (num - 1) if num > 1 else 0
        return [start + i * step for i in range(num)]
    
    def arange(self, start, stop=None, step=1):
        if stop is None:
            stop, start = start, 0
        return list(range(start, stop, step))

class MockRandom:
    """Mock numpy.random for testing."""
    
    def randn(self, *args):
        if len(args) == 0:
            return 0.0
        elif len(args) == 1:
            return [0.0] * args[0]
        elif len(args) == 2:
            return [[0.0] * args[1] for _ in range(args[0])]
        return 0.0
    
    def random(self, *args):
        if len(args) == 0:
            return 0.5
        elif len(args) == 1:
            return [0.5] * args[0]
        elif len(args) == 2:
            return [[0.5] * args[1] for _ in range(args[0])]
        return 0.5
    
    def uniform(self, low, high, size=None):
        if size is None:
            return (low + high) / 2
        elif isinstance(size, int):
            return [(low + high) / 2] * size
        elif len(size) == 2:
            return [[(low + high) / 2] * size[1] for _ in range(size[0])]
        return (low + high) / 2

# Mock sklearn
class MockSklearn:
    """Mock sklearn for testing."""
    
    def __init__(self):
        self.metrics = MockMetrics()
        self.model_selection = MockModelSelection()
        self.ensemble = MockEnsemble()
        self.linear_model = MockLinearModel()
        self.tree = MockTree()
        self.svm = MockSVM()
    
    class MockMetrics:
        def accuracy_score(self, y_true, y_pred):
            if not y_true or not y_pred:
                return 0.0
            correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
            return correct / len(y_true)
        
        def precision_score(self, y_true, y_pred, average='binary'):
            return 0.8  # Mock precision
        
        def recall_score(self, y_true, y_pred, average='binary'):
            return 0.8  # Mock recall
        
        def f1_score(self, y_true, y_pred, average='binary'):
            return 0.8  # Mock F1 score
        
        def roc_auc_score(self, y_true, y_pred):
            return 0.8  # Mock ROC AUC
        
        def confusion_matrix(self, y_true, y_pred):
            return [[10, 2], [2, 10]]  # Mock confusion matrix
        
        def classification_report(self, y_true, y_pred):
            return "Mock classification report"
        
        def matthews_corrcoef(self, y_true, y_pred):
            return 0.8  # Mock Matthews correlation coefficient
        
        def cohen_kappa_score(self, y_true, y_pred):
            return 0.8  # Mock Cohen's kappa
        
        def log_loss(self, y_true, y_pred):
            return 0.5  # Mock log loss
    
    class MockModelSelection:
        def cross_val_score(self, estimator, X, y, cv=5):
            return [0.8, 0.8, 0.8, 0.8, 0.8]  # Mock CV scores
        
        def train_test_split(self, X, y, test_size=0.2, random_state=None):
            split_idx = int(len(X) * (1 - test_size))
            return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
    
    class MockEnsemble:
        class RandomForestClassifier:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            
            def fit(self, X, y):
                return self
            
            def predict(self, X):
                return [0] * len(X)
            
            def predict_proba(self, X):
                return [[0.5, 0.5]] * len(X)
            
            def score(self, X, y):
                return 0.8
        
        class GradientBoostingClassifier:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            
            def fit(self, X, y):
                return self
            
            def predict(self, X):
                return [0] * len(X)
            
            def predict_proba(self, X):
                return [[0.5, 0.5]] * len(X)
            
            def score(self, X, y):
                return 0.8
    
    class MockLinearModel:
        class LogisticRegression:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            
            def fit(self, X, y):
                return self
            
            def predict(self, X):
                return [0] * len(X)
            
            def predict_proba(self, X):
                return [[0.5, 0.5]] * len(X)
            
            def score(self, X, y):
                return 0.8
    
    class MockTree:
        class DecisionTreeClassifier:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            
            def fit(self, X, y):
                return self
            
            def predict(self, X):
                return [0] * len(X)
            
            def predict_proba(self, X):
                return [[0.5, 0.5]] * len(X)
            
            def score(self, X, y):
                return 0.8
    
    class MockSVM:
        class SVC:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            
            def fit(self, X, y):
                return self
            
            def predict(self, X):
                return [0] * len(X)
            
            def predict_proba(self, X):
                return [[0.5, 0.5]] * len(X)
            
            def score(self, X, y):
                return 0.8

# Mock matplotlib
class MockMatplotlib:
    """Mock matplotlib for testing."""
    
    def __init__(self):
        self.pyplot = MockPyplot()
    
    class MockPyplot:
        def figure(self, **kwargs):
            return MockFigure()
        
        def plot(self, *args, **kwargs):
            return [MockLine2D()]
        
        def scatter(self, *args, **kwargs):
            return MockPathCollection()
        
        def hist(self, *args, **kwargs):
            return [MockRectangle()]
        
        def show(self):
            pass
        
        def savefig(self, filename, **kwargs):
            pass
        
        def title(self, title):
            pass
        
        def xlabel(self, label):
            pass
        
        def ylabel(self, label):
            pass
        
        def legend(self):
            pass
        
        def subplot(self, *args, **kwargs):
            return MockAxes()

class MockFigure:
    """Mock matplotlib Figure for testing."""
    pass

class MockLine2D:
    """Mock matplotlib Line2D for testing."""
    pass

class MockPathCollection:
    """Mock matplotlib PathCollection for testing."""
    pass

class MockRectangle:
    """Mock matplotlib Rectangle for testing."""
    pass

class MockAxes:
    """Mock matplotlib Axes for testing."""
    pass

# Mock seaborn
class MockSeaborn:
    """Mock seaborn for testing."""
    
    def heatmap(self, data, **kwargs):
        return MockAxes()
    
    def pairplot(self, data, **kwargs):
        return MockFigure()
    
    def distplot(self, data, **kwargs):
        return MockAxes()

# Install mocks
def install_mocks():
    """Install mock dependencies."""
    logger.info("Installing mock dependencies...")
    
    # Mock pandas
    sys.modules['pandas'] = type('MockPandas', (), {
        'DataFrame': MockDataFrame,
        'Series': MockSeries,
        '__version__': '1.5.0'
    })()
    
    # Mock numpy
    sys.modules['numpy'] = MockNumpy()
    
    # Mock sklearn
    sys.modules['sklearn'] = MockSklearn()
    sys.modules['sklearn.metrics'] = MockSklearn.MockMetrics()
    sys.modules['sklearn.model_selection'] = MockSklearn.MockModelSelection()
    sys.modules['sklearn.ensemble'] = MockSklearn.MockEnsemble()
    sys.modules['sklearn.linear_model'] = MockSklearn.MockLinearModel()
    sys.modules['sklearn.tree'] = MockSklearn.MockTree()
    sys.modules['sklearn.svm'] = MockSklearn.MockSVM()
    
    # Mock matplotlib
    sys.modules['matplotlib'] = MockMatplotlib()
    sys.modules['matplotlib.pyplot'] = MockMatplotlib.MockPyplot()
    
    # Mock seaborn
    sys.modules['seaborn'] = MockSeaborn()
    
    logger.info("✅ Mock dependencies installed successfully")

# Auto-install mocks when module is imported
try:
    install_mocks()
except Exception as e:
    logger.warning(f"Could not auto-install mocks: {e}")