import os
import json
import copy
from typing import Optional

from src.config import CONFIG
from src.utils.logger import system_logger
from src.analyst.analyst import Analyst
from src.tactician.tactician import Tactician
from src.strategist.strategist import Strategist
from src.supervisor.performance_reporter import PerformanceReporter # Import PerformanceReporter

class ModelManager:
    """
    Manages the loading, serving, and hot-swapping of trading models and parameters.
    This allows for updating the strategy without restarting the bot.
    """
    def __init__(self, firestore_manager=None, performance_reporter: Optional[PerformanceReporter] = None): # Added performance_reporter
        self.logger = system_logger.getChild('ModelManager')
        self.firestore_manager = firestore_manager
        self.performance_reporter = performance_reporter # Store reporter instance
        
        # These will hold the live, running instances of the modules
        self.analyst = None
        self.tactician = None
        self.strategist = None
        self.current_params = None
        
        # Load the initial 'champion' models on startup
        self.load_models(model_version="champion", performance_reporter=self.performance_reporter) # Pass reporter

    def load_models(self, model_version="champion", performance_reporter: Optional[PerformanceReporter] = None): # Added performance_reporter
        """
        Loads a specific version of the models (e.g., 'champion' or 'challenger').
        Instantiates the core logic modules with the appropriate models and parameters.
        """
        self.logger.info(f"Loading '{model_version}' model set...")
        
        # Determine the path for parameters based on model_version
        if model_version == "champion":
            params = CONFIG['best_params']
            self.logger.info("Using CONFIG['best_params'] as champion parameters.")
        elif model_version == "challenger":
            params_path = os.path.join("models/challenger", "optimized_params.json")
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    params = json.load(f)
                self.logger.info(f"Loaded challenger parameters from {params_path}")
            else:
                self.logger.warning(f"Challenger parameters file not found at {params_path}. Cannot load challenger.")
                return False # Indicate failure to load challenger
        else:
            self.logger.error(f"Unknown model version: {model_version}. Cannot load models.")
            return False

        self.current_params = params

        # Update CONFIG['best_params'] with the currently loaded parameters
        CONFIG['best_params'] = copy.deepcopy(self.current_params)
        
        # Instantiate the modules. Analyst will load its own sub-models (regime classifier, ensembles)
        self.analyst = Analyst(exchange_client=None, state_manager=None)
        self.strategist = Strategist(exchange_client=None, state_manager=None)
        # Pass performance_reporter to Tactician
        self.tactician = Tactician(exchange_client=None, state_manager=None, performance_reporter=performance_reporter)

        self.logger.info(f"'{model_version}' model set and modules are now loaded.")
        return True # Indicate successful loading

    def promote_challenger_to_champion(self):
        """
        Performs the hot-swap. Loads the 'challenger' models and replaces the
        live 'champion' instances without a restart.
        """
        self.logger.critical("--- HOT-SWAP: Promoting Challenger model to Champion ---")
        try:
            # When promoting, we need to ensure the performance_reporter is passed through
            # This requires a slight adjustment to how promote_challenger_to_champion is called
            # or how the ModelManager gets the performance_reporter.
            # For simplicity, we'll assume the live system will re-instantiate Tactician
            # with the correct reporter if a hot-swap happens.
            # For now, this method will just load the models.
            if self.load_models(model_version="challenger", performance_reporter=self.performance_reporter): # Pass reporter
                self.logger.critical("--- HOT-SWAP COMPLETE: System is now running on the new model. ---")
                return True
            else:
                self.logger.error("Failed to load challenger models. Promotion aborted.")
                return False
        except Exception as e:
            self.logger.error(f"Failed to promote challenger model: {e}", exc_info=True)
            return False

    def get_analyst(self):
        return self.analyst

    def get_strategist(self):
        return self.strategist

    def get_tactician(self, performance_reporter: Optional[PerformanceReporter] = None): # Allow passing reporter
        # If a reporter is passed, update the tactician's reporter
        if performance_reporter and self.tactician:
            self.tactician.performance_reporter = performance_reporter
        return self.tactician

