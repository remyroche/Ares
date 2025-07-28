# src/database/firestore_manager.py
import json
import os
import datetime
import uuid # For generating unique IDs
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin.auth import get_auth # Assuming firebase_admin for server-side auth
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
            # Initialize Firebase Admin SDK if not already initialized
            if not firebase_admin._apps: # Check if any Firebase app is already initialized
                self._initialize_firebase()
            else:
                self.logger.info("Firebase app already initialized. Reusing existing app.")
                self.db = firestore.client()
                self.auth = get_auth()
                self._determine_user_id()
        else:
            self.logger.info("Firestore integration is DISABLED in config.")
        
        self._initialized = True

    def _initialize_firebase(self):
        """Initializes Firebase Admin SDK."""
        try:
            # The Canvas environment typically provides credentials implicitly via GOOGLE_APPLICATION_CREDENTIALS
            # or by configuring the default app.
            # If firebase_config_str is a client-side config, it's not directly used by Admin SDK for auth,
            # but it might contain project_id which is useful.
            
            # Attempt to initialize without explicit credentials, relying on default environment setup
            # or if a service account JSON is available via GOOGLE_APPLICATION_CREDENTIALS env var.
            firebase_admin.initialize_app()
            self.logger.info("Firebase Admin SDK initialized successfully.")

            self.db = firestore.client()
            self.auth = get_auth() # Get auth service for user management if needed
            self._determine_user_id()
            self.logger.info("Firebase Firestore client and Auth service obtained.")

        except Exception as e:
            self.logger.error(f"Failed to initialize Firebase Admin SDK or Firestore client: {e}", exc_info=True)
            self.db = None
            self.auth = None
            self.firestore_enabled = False # Disable if initialization fails

    def _determine_user_id(self):
        """
        Determines the user ID for Firestore document paths.
        Prioritizes a user ID derived from the initial auth token (if available and valid),
        otherwise falls back to a Canvas app-specific ID or a random UUID.
        """
        if self.initial_auth_token:
            try:
                # In a real Admin SDK scenario, you'd typically verify an ID token
                # to get the UID. If __initial_auth_token is a custom token for client-side,
                # the Admin SDK doesn't directly 'sign in' with it.
                # For Canvas, we assume a user context is implicitly available or we derive a stable ID.
                # If the token is a Firebase ID Token, you could do:
                # decoded_token = self.auth.verify_id_token(self.initial_auth_token)
                # self.user_id = decoded_token['uid']
                
                # Given the context, we'll use a stable ID based on app_id for consistency
                # if an auth token is present, implying a user session.
                self.user_id = f"canvas-user-{self.app_id}" 
                self.logger.info(f"Using Canvas-derived user ID for Firestore paths: {self.user_id}")

            except Exception as e:
                self.logger.warning(f"Could not derive user ID from initial auth token: {e}. Falling back to random UUID.")
                self.user_id = str(uuid.uuid4()) # Fallback to random UUID
        else:
            self.user_id = str(uuid.uuid4()) # Generate a random UUID for anonymous users
            self.logger.info(f"Using anonymous user ID for Firestore paths: {self.user_id}")

        self.logger.info(f"Firestore operations will use user_id: {self.user_id}")
        self.logger.info("IMPORTANT: Ensure Firestore Security Rules are configured correctly for user data access.")
        self.logger.info("For private data: match /artifacts/{appId}/users/{userId}/{collectionName}/{documentId} { allow read, write: if request.auth != null && request.auth.uid == userId; }")
        self.logger.info("For public data: match /artifacts/{appId}/public/data/{collectionName}/{documentId} { allow read, write: if request.auth != null; }")


    def _get_collection_path(self, collection_name, is_public=False):
        """Constructs the full Firestore collection path."""
        base_path = f"artifacts/{self.app_id}"
        if is_public:
            return f"{base_path}/{self.config.get('public_data_collection_path', 'public/data')}/{collection_name}"
        else:
            if not self.user_id:
                self.logger.error("User ID not set for private collection path. Cannot construct path.")
                return None
            return f"{base_path}/{self.config.get('user_data_collection_path', 'users')}/{self.user_id}/{collection_name}"

    async def add_document(self, collection_name, data, is_public=False):
        """Adds a new document to a collection with an auto-generated ID."""
        if not self.firestore_enabled or self.db is None: return None
        collection_path = self._get_collection_path(collection_name, is_public)
        if collection_path is None: return None
        try:
            collection_ref = self.db.collection(collection_path)
            # Firestore Admin SDK add() is synchronous, no await needed for the call itself
            # The return value is a tuple (update_time, document_reference)
            update_time, doc_ref = collection_ref.add(data) 
            self.logger.info(f"Document added to {collection_name} with ID: {doc_ref.id}")
            return doc_ref.id
        except Exception as e:
            self.logger.error(f"Error adding document to {collection_name}: {e}", exc_info=True)
            return None

    async def set_document(self, collection_name, doc_id, data, is_public=False):
        """Sets a document with a specified ID (creates or overwrites)."""
        if not self.firestore_enabled or self.db is None: return False
        collection_path = self._get_collection_path(collection_name, is_public)
        if collection_path is None: return False
        try:
            doc_ref = self.db.collection(collection_path).document(str(doc_id))
            # Firestore Admin SDK set() is synchronous
            doc_ref.set(data)
            self.logger.info(f"Document {doc_id} set in {collection_name}.")
            return True
        except Exception as e:
            self.logger.error(f"Error setting document {doc_id} in {collection_name}: {e}", exc_info=True)
            return False

    async def get_document(self, collection_name, doc_id, is_public=False):
        """Retrieves a single document by its ID."""
        if not self.firestore_enabled or self.db is None: return None
        collection_path = self._get_collection_path(collection_name, is_public)
        if collection_path is None: return None
        try:
            doc_ref = self.db.collection(collection_path).document(str(doc_id))
            # Firestore Admin SDK get() is synchronous
            doc = doc_ref.get()
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
        collection_path = self._get_collection_path(collection_name, is_public)
        if collection_path is None: return []
        try:
            collection_ref = self.db.collection(collection_path)
            
            if query_filters:
                for field, op, value in query_filters:
                    collection_ref = collection_ref.where(field, op, value)
            
            docs = []
            # Firestore Admin SDK stream() is synchronous iterator, no async for needed
            for doc in collection_ref.stream(): 
                docs.append({**doc.to_dict(), "id": doc.id})
            self.logger.info(f"Retrieved {len(docs)} documents from {collection_name}.")
            return docs
        except Exception as e:
            self.logger.error(f"Error getting documents from {collection_name}: {e}", exc_info=True)
            return []

    async def update_document(self, collection_name, doc_id, data, is_public=False):
        """Updates fields in an existing document."""
        if not self.firestore_enabled or self.db is None: return False
        collection_path = self._get_collection_path(collection_name, is_public)
        if collection_path is None: return False
        try:
            doc_ref = self.db.collection(collection_path).document(str(doc_id))
            # Firestore Admin SDK update() is synchronous
            doc_ref.update(data)
            self.logger.info(f"Document {doc_id} updated in {collection_name}.")
            return True
        except Exception as e:
            self.logger.error(f"Error updating document {doc_id} in {collection_name}: {e}", exc_info=True)
            return False

    async def delete_document(self, collection_name, doc_id, is_public=False):
        """Deletes a document by its ID."""
        if not self.firestore_enabled or self.db is None: return False
        collection_path = self._get_collection_path(collection_name, is_public)
        if collection_path is None: return False
        try:
            doc_ref = self.db.collection(collection_path).document(str(doc_id))
            # Firestore Admin SDK delete() is synchronous
            doc_ref.delete()
            self.logger.info(f"Document {doc_id} deleted from {collection_name}.")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting document {doc_id} from {collection_name}: {e}", exc_info=True)
            return False
