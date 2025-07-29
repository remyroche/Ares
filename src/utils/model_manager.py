# src/utils/model_manager.py
import os
import json
import copy
from src.config import CONFIG
from src.utils.logger import system_logger
from src.analyst.analyst import Analyst
from src.tactician.tactician import Tactician # Assuming Tactician is still needed here
from src.strategist.strategist import Strategist # Assuming Strategist is still needed here

class ModelManager:
    """
    Manages the loading, serving, and hot-swapping of trading models and parameters.
    This allows for updating the strategy without restarting the bot.
    """
    def __init__(self, firestore_manager=None):
        self.logger = system_logger.getChild('ModelManager')
        self.firestore_manager = firestore_manager
        
        # These will hold the live, running instances of the modules
        self.analyst = None
        self.tactician = None
        self.strategist = None
        self.current_params = None
        
        # Load the initial 'champion' models on startup
        self.load_models(model_version="champion")

    def load_models(self, model_version="champion"):
        """
        Loads a specific version of the models (e.g., 'champion' or 'challenger').
        Instantiates the core logic modules with the appropriate models and parameters.
        """
        self.logger.info(f"Loading '{model_version}' model set...")
        
        # Determine the path for parameters based on model_version
        if model_version == "champion":
            # Champion parameters are the CONFIG['best_params'] which are updated by optimization
            params = CONFIG['best_params']
            self.logger.info("Using CONFIG['best_params'] as champion parameters.")
        elif model_version == "challenger":
            # Challenger parameters are saved to a specific file after training pipeline
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
        # This ensures all components that rely on CONFIG['best_params'] get the correct values
        # for the active champion/challenger.
        CONFIG['best_params'] = copy.deepcopy(self.current_params)
        
        # Instantiate the modules. Analyst will load its own sub-models (regime classifier, ensembles)
        # based on the 'final' fold_id if in live mode.
        # Note: The Analyst, Strategist, and Tactician constructors need the exchange_client and state_manager
        # in a real running system. For now, we pass CONFIG and firestore_manager.
        # This part of the ModelManager might need adjustment based on how it's used in main_launcher.py
        # For the purpose of training pipeline, it's mostly about setting CONFIG['best_params'].

        # Assuming these are for live operation, they would be instantiated with actual clients
        # For training pipeline context, they might not be fully functional here.
        # Re-instantiate them to ensure they pick up the new CONFIG['best_params']
        self.analyst = Analyst(exchange_client=None, state_manager=None) # Pass None for clients as they are not used in init
        self.strategist = Strategist(exchange_client=None, state_manager=None) # Pass None for clients
        self.tactician = Tactician(exchange_client=None, state_manager=None) # Pass None for clients

        self.logger.info(f"'{model_version}' model set and modules are now loaded.")
        return True # Indicate successful loading

    def promote_challenger_to_champion(self):
        """
        Performs the hot-swap. Loads the 'challenger' models and replaces the
        live 'champion' instances without a restart.
        """
        self.logger.critical("--- HOT-SWAP: Promoting Challenger model to Champion ---")
        try:
            # Load the challenger models and parameters
            if self.load_models(model_version="challenger"):
                # After loading, the new instances are now in self.analyst, etc.
                # The main pipeline, which holds a reference to the ModelManager instance,
                # will automatically start using these new module instances on its next loop.
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

    def get_tactician(self):
        return self.tactician
