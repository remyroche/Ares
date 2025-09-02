
# REFACTORING PATTERN: Strategy Pattern
# For: DataCollectionStep._log_detailed_data_extract (Complexity: 41)

# BEFORE: Giant method with many conditional branches
def _log_detailed_data_extract(self, data_dict):
    # Huge if-elif chain handling different data types
    if data_type == "klines":
        # 50 lines of klines logging
    elif data_type == "aggtrades":
        # 50 lines of aggtrades logging
    elif data_type == "futures":
        # 50 lines of futures logging
    ...

# AFTER: Strategy pattern with dedicated handlers
class DataLoggerStrategy:
    """Abstract base for data logging strategies"""
    def log(self, data): 
        raise NotImplementedError

class KlinesLogger(DataLoggerStrategy):
    def log(self, data):
        # Focused klines logging logic
        ...

class AggtradesLogger(DataLoggerStrategy):
    def log(self, data):
        # Focused aggtrades logging logic
        ...

def _log_detailed_data_extract(self, data_dict):
    """Simplified method using strategy pattern"""
    loggers = {
        'klines': KlinesLogger(),
        'aggtrades': AggtradesLogger(),
        'futures': FuturesLogger()
    }
    
    for data_type, data in data_dict.items():
        logger = loggers.get(data_type)
        if logger:
            logger.log(data)
