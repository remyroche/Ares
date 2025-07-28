# src/utils/model_manager.py
import os
import json
import copy
from src.config import CONFIG
from src.utils.logger import system_logger
from src.analyst.analyst import Analyst
from src.tactician.tactician import Tactician
from src.strategist.strategist import Strategist

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
        
        self.load_models()

    def load_models(self, model_version="champion"):
        """
        Loads a specific version of the models (e.g., 'champion' or 'challenger').
        Instantiates the core logic modules with the appropriate models and parameters.
        """
        self.logger.info(f"Loading '{model_version}' model set...")
        
        model_path = "models/analyst" if model_version == "champion" else "models/challenger"
        params_path = os.path.join(model_path, "optimized_params.json")

        # Load the parameters
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                params = json.load(f)
            self.logger.info(f"Loaded parameters from {params_path}")
        else:
            self.logger.warning(f"Parameters file not found at {params_path}. Using default from config.")
            params = CONFIG['BEST_PARAMS']
        
        self.current_params = params

        # Create a temporary config with the loaded parameters
        temp_config = copy.deepcopy(CONFIG)
        temp_config['BEST_PARAMS'] = self.current_params
        
        # Instantiate the modules with the specific config
        # We assume the Analyst loads its own sub-models based on paths in the config
        self.analyst = Analyst(config=temp_config, firestore_manager=self.firestore_manager)
        self.strategist = Strategist(config=temp_config)
        self.tactician = Tactician(config=temp_config, firestore_manager=self.firestore_manager)

        self.logger.info(f"'{model_version}' model set and modules are now loaded.")

    def promote_challenger_to_champion(self):
        """
        Performs the hot-swap. Loads the 'challenger' models and replaces the
        live 'champion' instances without a restart.
        """
        self.logger.critical("--- HOT-SWAP: Promoting Challenger model to Champion ---")
        try:
            # Load the challenger models into the manager's attributes
            self.load_models(model_version="challenger")
            
            # After loading, the new instances are now in self.analyst, etc.
            # The main pipeline, which holds a reference to the ModelManager instance,
            # will automatically start using these new module instances on its next loop.
            
            self.logger.critical("--- HOT-SWAP COMPLETE: System is now running on the new model. ---")
            return True
        except Exception as e:
            self.logger.error(f"Failed to promote challenger model: {e}", exc_info=True)
            return False
