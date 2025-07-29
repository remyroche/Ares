import asyncio
import os
import uuid
from functools import partial
from typing import Any, Dict, List, Optional, Callable, Tuple


import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin.auth import get_auth
from loguru import logger

from src.config import settings, CONFIG # Import CONFIG

class FirestoreManager:
    """
    Manages all interactions with Google Cloud Firestore.
    This version uses the firebase-admin SDK (which is synchronous) and adapts it
    for an asyncio application by running blocking calls in a thread pool.
    Its functionality can be conditionally disabled based on CONFIG['DATABASE_TYPE'].
    """

    def __init__(self):
        self._db: Optional[firestore.Client] = None # Fixed: Type hint
        self._auth: Any = None # Fixed: Type hint
        self._user_id: Optional[str] = None # Fixed: Type hint
        self._app_id: Optional[str] = None # Fixed: Type hint
        self._initialized = False
        self._firestore_enabled = False # This will be set based on config

    async def initialize(self):
        """Asynchronously initializes the Firestore connection."""
        if self._initialized:
            logger.info("FirestoreManager already initialized.")
            return

        # Check if Firestore is enabled in the settings AND if DATABASE_TYPE is 'firestore'
        if not settings.google_application_credentials or not settings.firestore_project_id or CONFIG.get("DATABASE_TYPE") != "firestore":
            logger.warning("Firestore not configured or DATABASE_TYPE is not 'firestore'. Firestore operations will be disabled.")
            self._firestore_enabled = False
            self._initialized = True
            return
        
        self._firestore_enabled = True
        self._app_id = os.environ.get('__app_id', 'default-ares-app-id')
        initial_auth_token = os.environ.get('__initial_auth_token')

        try:
            loop = asyncio.get_running_loop()
            # Run blocking initialization in a separate thread
            await loop.run_in_executor(None, self._blocking_initialize)

            # Determine user ID (this part is not blocking)
            self._determine_user_id(initial_auth_token)

            self._initialized = True
            logger.info("FirestoreManager initialized successfully.")
            logger.info(f"Firestore operations will use user_id: {self._user_id}") # Fixed: Use _user_id
            logger.info("Ensure Firestore Security Rules are configured for user data access.")

        except Exception as e:
            logger.error(f"Failed to initialize FirestoreManager: {e}", exc_info=True)
            self._db = None
            self._auth = None
            self._firestore_enabled = False

    def _blocking_initialize(self):
        """Synchronous part of the initialization. Runs in a thread pool."""
        if not firebase_admin._apps:
            cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred, {'projectId': settings.firestore_project_id})
            logger.info("Firebase Admin SDK initialized.")
        else:
            logger.info("Firebase app already initialized. Reusing existing app.")

        self._db = firestore.client()
        self._auth = get_auth()

    def _determine_user_id(self, initial_auth_token: Optional[str]): # Fixed: Type hint
        """Determines the user ID for Firestore document paths."""
        if initial_auth_token:
            try:
                self._user_id = f"canvas-user-{self._app_id}"
                logger.info(f"Using Canvas-derived user ID for Firestore paths: {self._user_id}")
            except Exception as e:
                logger.warning(f"Could not derive user ID from auth token: {e}. Falling back.")
                self._user_id = str(uuid.uuid4())
        else:
            self._user_id = str(uuid.uuid4())
            logger.info(f"Using anonymous user ID for Firestore paths: {self._user_id}")

    def _get_collection_path(self, collection_name: str, is_public: bool = False) -> Optional[str]: # Fixed: Return type hint
        """Constructs the full Firestore collection path."""
        if self._app_id is None: # Fixed: Check if _app_id is None
            logger.error("App ID not set. Cannot construct collection path.")
            return None
        
        base_path = f"artifacts/{self._app_id}"
        if is_public:
            return f"{base_path}/public/data/{collection_name}"
        else:
            if not self._user_id: # Fixed: Use _user_id
                logger.error("User ID not set. Cannot construct private collection path.")
                return None
            return f"{base_path}/users/{self._user_id}/{collection_name}"

    async def _execute_blocking(self, func: Callable, *args: Any, **kwargs: Any) -> Any: # Fixed: Type hints
        """Helper to run any blocking function in a thread pool."""
        if not self._firestore_enabled or not self._initialized or not self._db:
            logger.warning("Firestore not available. Cannot perform operation.")
            return None
        loop = asyncio.get_running_loop()
        p_func = partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, p_func)

    async def set_document(self, collection_name: str, doc_id: str, data: Dict[str, Any], is_public: bool = False) -> bool: # Fixed: Type hints
        """Sets a document with a specified ID (creates or overwrites)."""
        if not self._firestore_enabled: # Fixed: Check if enabled
            self.logger.debug(f"Firestore disabled. Skipping set_document for {collection_name}/{doc_id}.")
            return False

        collection_path = self._get_collection_path(collection_name, is_public)
        if not collection_path: return False

        def _blocking_op():
            if self._db: # Fixed: Check if _db is not None
                doc_ref = self._db.collection(collection_path).document(str(doc_id))
                doc_ref.set(data)
            else:
                raise RuntimeError("Firestore client not available.")
        
        try:
            await self._execute_blocking(_blocking_op)
            logger.info(f"Document {doc_id} set in {collection_name}.")
            return True
        except Exception as e:
            logger.error(f"Error setting document {doc_id} in {collection_name}: {e}", exc_info=True)
            return False

    async def get_document(self, collection_name: str, doc_id: str, is_public: bool = False) -> Optional[Dict[str, Any]]: # Fixed: Type hints
        """Retrieves a single document by its ID."""
        if not self._firestore_enabled: # Fixed: Check if enabled
            self.logger.debug(f"Firestore disabled. Skipping get_document for {collection_name}/{doc_id}.")
            return None

        collection_path = self._get_collection_path(collection_name, is_public)
        if not collection_path: return None

        def _blocking_op():
            if self._db: # Fixed: Check if _db is not None
                doc_ref = self._db.collection(collection_path).document(str(doc_id))
                doc = doc_ref.get()
                return doc.to_dict() if doc.exists else None
            else:
                raise RuntimeError("Firestore client not available.")
        
        try:
            result = await self._execute_blocking(_blocking_op)
            if result is not None:
                logger.debug(f"Document {doc_id} retrieved from {collection_name}.")
            else:
                logger.warning(f"Document {doc_id} not found in {collection_name}.")
            return result
        except Exception as e:
            logger.error(f"Error getting document {doc_id} from {collection_name}: {e}", exc_info=True)
            return None

    async def add_document(self, collection_name: str, data: Dict[str, Any], is_public: bool = False) -> Optional[str]: # Fixed: Return type hint
        """Adds a document with an auto-generated ID."""
        if not self._firestore_enabled: # Fixed: Check if enabled
            self.logger.debug(f"Firestore disabled. Skipping add_document for {collection_name}.")
            return None

        collection_path = self._get_collection_path(collection_name, is_public)
        if not collection_path: return None

        def _blocking_op():
            if self._db: # Fixed: Check if _db is not None
                doc_ref = self._db.collection(collection_path).add(data)
                return doc_ref[1].id # Return the ID of the newly created document
            else:
                raise RuntimeError("Firestore client not available.")

        try:
            doc_id = await self._execute_blocking(_blocking_op)
            logger.info(f"Document added to {collection_name} with ID: {doc_id}.")
            return doc_id
        except Exception as e:
            logger.error(f"Error adding document to {collection_name}: {e}", exc_info=True)
            return None


    async def get_collection(self, collection_name: str, is_public: bool = False, query_filters: Optional[List[Tuple[str, str, Any]]] = None) -> List[Dict[str, Any]]: # Fixed: Type hints
        """Retrieves all documents from a collection, optionally with filters."""
        if not self._firestore_enabled: # Fixed: Check if enabled
            self.logger.debug(f"Firestore disabled. Skipping get_collection for {collection_name}.")
            return []

        collection_path = self._get_collection_path(collection_name, is_public)
        if not collection_path: return []

        def _blocking_op():
            if self._db: # Fixed: Check if _db is not None
                collection_ref = self._db.collection(collection_path)
                if query_filters:
                    for field, op, value in query_filters:
                        collection_ref = collection_ref.where(field, op, value)
                return [{**doc.to_dict(), "id": doc.id} for doc in collection_ref.stream()]
            else:
                raise RuntimeError("Firestore client not available.")

        try:
            docs = await self._execute_blocking(_blocking_op)
            logger.info(f"Retrieved {len(docs)} documents from {collection_name}.")
            return docs
        except Exception as e:
            logger.error(f"Error getting documents from {collection_name}: {e}", exc_info=True)
            return []

    async def delete_document(self, collection_name: str, doc_id: str, is_public: bool = False) -> bool: # Fixed: Return type hint
        """Deletes a document by its ID."""
        if not self._firestore_enabled: # Fixed: Check if enabled
            self.logger.debug(f"Firestore disabled. Skipping delete_document for {collection_name}/{doc_id}.")
            return False

        collection_path = self._get_collection_path(collection_name, is_public)
        if not collection_path: return False

        def _blocking_op():
            if self._db: # Fixed: Check if _db is not None
                self._db.collection(collection_path).document(str(doc_id)).delete()
            else:
                raise RuntimeError("Firestore client not available.")

        try:
            await self._execute_blocking(_blocking_op)
            logger.info(f"Document {doc_id} deleted from {collection_name}.")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id} from {collection_name}: {e}", exc_info=True)
            return False

# Global database instance.
# It should be initialized asynchronously in the main application entry point.
db_manager = FirestoreManager()
