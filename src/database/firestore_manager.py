# src/database/firestore_manager.py
import json
import os
import datetime
import uuid # For generating unique IDs
from firebase_admin import credentials, initialize_app, firestore
from firebase_admin.auth import signInWithCustomToken, signInAnonymously, get_auth # Assuming firebase_admin for server-side auth
from utils.logger import system_logger

class FirestoreManager:
    """
    Manages all interactions with Google Cloud Firestore.
    Handles Firebase initialization, authentication, and CRUD operations.
    """
    _instance = None # Singleton instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FirestoreManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config=None, app_id=None, firebase_config_str=None, initial_auth_token=None):
        if self._initialized:
            return

        self.logger = system_logger.getChild('FirestoreManager')
        self.config = config.get("firestore", {}) if config else {}
        self.app_id = app_id if app_id else (os.environ.get('__app_id') or 'default-ares-app-id')
        self.firebase_config_str = firebase_config_str if firebase_config_str else os.environ.get('__firebase_config')
        self.initial_auth_token = initial_auth_token if initial_auth_token else os.environ.get('__initial_auth_token')

        self.db = None
        self.auth = None
        self.user_id = None
        self.firestore_enabled = self.config.get("enabled", False)

        if self.firestore_enabled:
            self._initialize_firebase()
        else:
            self.logger.info("Firestore integration is DISABLED in config.")
        
        self._initialized = True

    def _initialize_firebase(self):
        """Initializes Firebase Admin SDK and authenticates."""
        if self.db is not None:
            self.logger.info("Firebase already initialized.")
            return

        try:
            # Firebase Admin SDK requires credentials.
            # In a server environment (like Canvas), __firebase_config is usually provided
            # and might be a dictionary for client-side SDK.
            # For Python Admin SDK, it typically expects a service account key file.
            # If __firebase_config is a client-side config, we need to adapt.

            # Assuming __firebase_config is a JSON string of the client-side config
            # and we are running in an environment where the Admin SDK can infer credentials
            # or if a service account file is available.
            # For Canvas, the environment should handle it automatically if initialize_app is called.
            
            # If running locally, you might need a service account key file:
            # cred = credentials.Certificate("path/to/your/serviceAccountKey.json")
            # initialize_app(cred)

            # For Canvas environment, initialize_app() without args often works
            # if GOOGLE_APPLICATION_CREDENTIALS env var is set or default creds are available.
            # If __firebase_config is meant for client-side, this part needs careful handling.
            
            # Let's try to initialize with the provided config string if it's a valid JSON.
            # The Admin SDK's initialize_app() usually takes a dict.
            if self.firebase_config_str:
                firebase_config_dict = json.loads(self.firebase_config_str)
                # Check if an app is already initialized to avoid ValueError
                if not initialize_app(): # Check if default app exists
                    initialize_app(firebase_config_dict)
                else:
                    self.logger.info("Default Firebase app already initialized.")
            else:
                # Attempt to initialize without args, relying on default credentials
                if not initialize_app():
                    initialize_app()
                else:
                    self.logger.info("Default Firebase app already initialized.")

            self.db = firestore.client()
            self.auth = get_auth() # Get auth service

            self._authenticate_user()
            self.logger.info("Firebase Firestore initialized and authenticated.")

        except Exception as e:
            self.logger.error(f"Failed to initialize Firebase Firestore: {e}", exc_info=True)
            self.db = None
            self.auth = None
            self.firestore_enabled = False # Disable if initialization fails

    def _authenticate_user(self):
        """Authenticates the user using custom token or anonymously."""
        try:
            if self.initial_auth_token:
                # Use Admin SDK to verify/decode the token, not sign in with it directly
                # The signInWithCustomToken is for client SDK. For Admin SDK, we just need the UID.
                # If the token is for *this* Admin SDK instance to use, it's more complex.
                # Assuming the token is for a client, and we need a UID for document paths.
                # For Admin SDK, we typically operate with service account creds directly.
                # If the goal is to get the current user's UID for paths:
                # This part needs clarification based on how __initial_auth_token is used.
                
                # For Canvas, the `__initial_auth_token` is typically used with client-side SDK.
                # For Python Admin SDK, we usually operate with service account credentials
                # which give full admin access, and then enforce security rules based on `request.auth.uid`.
                # If we need a `user_id` for document paths, and it's tied to the user's session:
                # We can either derive it from the token (if it's a JWT) or generate a random one.

                # Let's simulate getting a user ID. In a real Admin SDK scenario,
                # you'd either have a specific user context or generate a UUID.
                # For the purpose of pathing, let's use a consistent ID.
                self.user_id = f"canvas-user-{self.app_id}" # Consistent ID for demo
                self.logger.info(f"Using derived user ID for Firestore paths: {self.user_id}")
            else:
                self.user_id = str(uuid.uuid4()) # Generate a random UUID for anonymous users
                self.logger.info(f"Using anonymous user ID for Firestore paths: {self.user_id}")

        except Exception as e:
            self.logger.error(f"Firebase authentication failed: {e}", exc_info=True)
            self.user_id = str(uuid.uuid4()) # Fallback to random UUID
            self.logger.warning(f"Falling back to random user ID: {self.user_id}")

    def _get_collection_path(self, collection_name, is_public=False):
        """Constructs the full Firestore collection path."""
        base_path = f"artifacts/{self.app_id}"
        if is_public:
            return f"{base_path}/{self.config.get('public_data_collection_path', 'public/data')}/{collection_name}"
        else:
            if not self.user_id:
                self.logger.error("User ID not set for private collection path.")
                return None
            return f"{base_path}/{self.config.get('user_data_collection_path', 'users')}/{self.user_id}/{collection_name}"

    async def add_document(self, collection_name, data, is_public=False):
        """Adds a new document to a collection with an auto-generated ID."""
        if not self.firestore_enabled or self.db is None: return None
        try:
            collection_ref = self.db.collection(self._get_collection_path(collection_name, is_public))
            doc_ref = await collection_ref.add(data) # Use await for async operations
            self.logger.info(f"Document added to {collection_name} with ID: {doc_ref[1].id}")
            return doc_ref[1].id
        except Exception as e:
            self.logger.error(f"Error adding document to {collection_name}: {e}", exc_info=True)
            return None

    async def set_document(self, collection_name, doc_id, data, is_public=False):
        """Sets a document with a specified ID (creates or overwrites)."""
        if not self.firestore_enabled or self.db is None: return False
        try:
            doc_ref = self.db.collection(self._get_collection_path(collection_name, is_public)).document(str(doc_id))
            await doc_ref.set(data)
            self.logger.info(f"Document {doc_id} set in {collection_name}.")
            return True
        except Exception as e:
            self.logger.error(f"Error setting document {doc_id} in {collection_name}: {e}", exc_info=True)
            return False

    async def get_document(self, collection_name, doc_id, is_public=False):
        """Retrieves a single document by its ID."""
        if not self.firestore_enabled or self.db is None: return None
        try:
            doc_ref = self.db.collection(self._get_collection_path(collection_name, is_public)).document(str(doc_id))
            doc = await doc_ref.get()
            if doc.exists:
                self.logger.info(f"Document {doc_id} retrieved from {collection_name}.")
                return doc.to_dict()
            else:
                self.logger.warning(f"Document {doc_id} not found in {collection_name}.")
                return None
        except Exception as e:
            self.logger.error(f"Error getting document {doc_id} from {collection_name}: {e}", exc_info=True)
            return None

    async def get_collection(self, collection_name, is_public=False, query_filters=None):
        """Retrieves all documents from a collection, optionally with filters."""
        if not self.firestore_enabled or self.db is None: return []
        try:
            collection_ref = self.db.collection(self._get_collection_path(collection_name, is_public))
            
            if query_filters:
                for field, op, value in query_filters:
                    collection_ref = collection_ref.where(field, op, value)
            
            docs = []
            async for doc in collection_ref.stream(): # Use async for stream
                docs.append({**doc.to_dict(), "id": doc.id})
            self.logger.info(f"Retrieved {len(docs)} documents from {collection_name}.")
            return docs
        except Exception as e:
            self.logger.error(f"Error getting documents from {collection_name}: {e}", exc_info=True)
            return []

    async def update_document(self, collection_name, doc_id, data, is_public=False):
        """Updates fields in an existing document."""
        if not self.firestore_enabled or self.db is None: return False
        try:
            doc_ref = self.db.collection(self._get_collection_path(collection_name, is_public)).document(str(doc_id))
            await doc_ref.update(data)
            self.logger.info(f"Document {doc_id} updated in {collection_name}.")
            return True
        except Exception as e:
            self.logger.error(f"Error updating document {doc_id} in {collection_name}: {e}", exc_info=True)
            return False

    async def delete_document(self, collection_name, doc_id, is_public=False):
        """Deletes a document by its ID."""
        if not self.firestore_enabled or self.db is None: return False
        try:
            doc_ref = self.db.collection(self._get_collection_path(collection_name, is_public)).document(str(doc_id))
            await doc_ref.delete()
            self.logger.info(f"Document {doc_id} deleted from {collection_name}.")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting document {doc_id} from {collection_name}: {e}", exc_info=True)
            return False

# Example of how to get the FirestoreManager instance
# firestore_manager = FirestoreManager()
# Make sure to pass __app_id, __firebase_config, __initial_auth_token
# when initializing in the main pipeline.
