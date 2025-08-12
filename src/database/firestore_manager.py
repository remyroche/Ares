import asyncio
import os
import uuid
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import firebase_admin
from firebase_admin import auth, credentials, firestore

from src.config import CONFIG, get_environment_settings  # Import CONFIG
from src.utils.error_handler import (
    ErrorRecoveryStrategies,
    error_context,
    handle_errors,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    missing,
    warning,
)

if TYPE_CHECKING:
    import logging


class FirestoreManager:
    """
    Manages all interactions with Google Cloud Firestore.
    This version uses the firebase-admin SDK (which is synchronous) and adapts it
    for an asyncio application by running blocking calls in a thread pool.
    Its functionality can be conditionally disabled based on CONFIG['DATABASE_TYPE'].
    """

    def __init__(self):
        self.logger: logging.Logger = system_logger.getChild("FirestoreManager")
        self._db: firestore.Client | None = None  # Fixed: Type hint
        self._auth: Any = None  # Fixed: Type hint
        self._user_id: str | None = None  # Fixed: Type hint
        self._app_id: str | None = None  # Fixed: Type hint
        self._initialized = False
        self._firestore_enabled = False  # This will be set based on config

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="firestore_initialization",
    )
    async def initialize(self):
        """Asynchronously initializes the Firestore connection."""
        if self._initialized:
            self.logger.info("FirestoreManager already initialized.")
            return

        # Check if Firestore is enabled in the settings AND if DATABASE_TYPE is 'firestore'
        with error_context("firestore_config_validation"):
            env_settings = get_environment_settings()
            if (
                not env_settings.google_application_credentials
                or not env_settings.firestore_project_id
                or CONFIG.get("DATABASE_TYPE") != "firestore"
            ):
                self.logger.warning(
                    "Firestore not configured or DATABASE_TYPE is not 'firestore'. Firestore operations will be disabled.",
                )
                self._firestore_enabled = False
                self._initialized = True
                return

        self._firestore_enabled = True
        self._app_id = ErrorRecoveryStrategies.safe_dict_access(
            os.environ,
            "__app_id",
            "default-ares-app-id",
        )
        initial_auth_token = ErrorRecoveryStrategies.safe_dict_access(
            os.environ,
            "__initial_auth_token",
            None,
        )

        with error_context("firestore_connection_setup"):
            loop = asyncio.get_running_loop()
            # Run blocking initialization in a separate thread
            await loop.run_in_executor(None, self._blocking_initialize)

            # Determine user ID (this part is not blocking)
            self._determine_user_id(initial_auth_token)

            self._initialized = True
            self.logger.info("FirestoreManager initialized successfully.")
            self.logger.info(f"Firestore operations will use user_id: {self._user_id}")
            self.logger.info(
                "Ensure Firestore Security Rules are configured for user data access.",
            )

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="firebase_blocking_initialization",
    )
    def _blocking_initialize(self):
        """Synchronous part of the initialization. Runs in a thread pool."""
        if not firebase_admin._apps:
            cred = credentials.ApplicationDefault()
            env_settings = get_environment_settings()
            firebase_admin.initialize_app(
                cred,
                {"projectId": env_settings.firestore_project_id},
            )
            self.logger.info("Firebase Admin SDK initialized.")
        else:
            self.logger.info(
                "Firebase app already initialized. Reusing existing app.",
            )

        self._db = firestore.client()
        self._auth = auth

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="user_id_determination",
    )
    def _determine_user_id(self, initial_auth_token: str | None):
        """Determines the user ID for Firestore document paths."""
        if initial_auth_token:
            self._user_id = f"canvas-user-{self._app_id}"
            self.logger.info(
                f"Using Canvas-derived user ID for Firestore paths: {self._user_id}",
            )
        else:
            self._user_id = str(uuid.uuid4())
            self.logger.info(
                f"Using anonymous user ID for Firestore paths: {self._user_id}",
            )

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="collection_path_construction",
    )
    def _get_collection_path(
        self,
        collection_name: str,
        is_public: bool = False,
    ) -> str | None:
        """Constructs the full Firestore collection path."""
        if self._app_id is None:
            self.print(error("App ID not set. Cannot construct collection path."))
            return None

        base_path = f"artifacts/{self._app_id}"
        if is_public:
            return f"{base_path}/public/data/{collection_name}"
        if not self._user_id:
            self.logger.error(
                "User ID not set. Cannot construct private collection path.",
            )
            return None
        return f"{base_path}/users/{self._user_id}/{collection_name}"

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="firestore_blocking_execution",
    )
    async def _execute_blocking(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Helper to run any blocking function in a thread pool."""
        if not self._firestore_enabled or not self._initialized or not self._db:
            self.print(warning("Firestore not available. Cannot perform operation."))
            return None

        loop = asyncio.get_running_loop()
        p_func = partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, p_func)

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="firestore_set_document",
    )
    async def set_document(
        self,
        collection_name: str,
        doc_id: str,
        data: dict[str, Any],
        is_public: bool = False,
    ) -> bool:
        """Sets a document with a specified ID (creates or overwrites)."""
        if not self._firestore_enabled:
            self.logger.debug(
                f"Firestore disabled. Skipping set_document for {collection_name}/{doc_id}.",
            )
            return False

        collection_path = self._get_collection_path(collection_name, is_public)
        if not collection_path:
            return False

        def _blocking_op():
            if self._db:
                doc_ref = self._db.collection(collection_path).document(str(doc_id))
                doc_ref.set(data)
            else:
                msg = "Firestore client not available."
                raise RuntimeError(msg)

        result = await self._execute_blocking(_blocking_op)
        if result is not None:
            self.logger.info(f"Document {doc_id} set in {collection_name}.")
            return True
        return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="firestore_get_document",
    )
    async def get_document(
        self,
        collection_name: str,
        doc_id: str,
        is_public: bool = False,
    ) -> dict[str, Any] | None:
        """Retrieves a single document by its ID."""
        if not self._firestore_enabled:
            self.logger.debug(
                f"Firestore disabled. Skipping get_document for {collection_name}/{doc_id}.",
            )
            return None

        collection_path = self._get_collection_path(collection_name, is_public)
        if not collection_path:
            return None

        def _blocking_op():
            if self._db:
                doc_ref = self._db.collection(collection_path).document(str(doc_id))
                doc = doc_ref.get()
                return doc.to_dict() if doc.exists else None
            msg = "Firestore client not available."
            raise RuntimeError(msg)

        result = await self._execute_blocking(_blocking_op)
        if result is not None:
            self.logger.debug(f"Document {doc_id} retrieved from {collection_name}.")
        else:
            self.print(missing("Document {doc_id} not found in {collection_name}."))
        return result

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="firestore_add_document",
    )
    async def add_document(
        self,
        collection_name: str,
        data: dict[str, Any],
        is_public: bool = False,
    ) -> str | None:
        """Adds a document with an auto-generated ID."""
        if not self._firestore_enabled:
            self.logger.debug(
                f"Firestore disabled. Skipping add_document for {collection_name}.",
            )
            return None

        collection_path = self._get_collection_path(collection_name, is_public)
        if not collection_path:
            return None

        def _blocking_op():
            if self._db:
                doc_ref = self._db.collection(collection_path).add(data)
                return doc_ref[1].id  # Return the ID of the newly created document
            msg = "Firestore client not available."
            raise RuntimeError(msg)

        doc_id = await self._execute_blocking(_blocking_op)
        if doc_id:
            self.logger.info(f"Document added to {collection_name} with ID: {doc_id}.")
        return doc_id

    @handle_errors(
        exceptions=(Exception,),
        default_return=[],
        context="firestore_get_collection",
    )
    async def get_collection(
        self,
        collection_name: str,
        is_public: bool = False,
        query_filters: list[tuple[str, str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieves all documents from a collection, optionally with filters."""
        if not self._firestore_enabled:
            self.logger.debug(
                f"Firestore disabled. Skipping get_collection for {collection_name}.",
            )
            return []

        collection_path = self._get_collection_path(collection_name, is_public)
        if not collection_path:
            return []

        def _blocking_op():
            if self._db:
                collection_ref = self._db.collection(collection_path)
                if query_filters:
                    for field, op, value in query_filters:
                        collection_ref = collection_ref.where(field, op, value)
                return [
                    {**doc.to_dict(), "id": doc.id} for doc in collection_ref.stream()
                ]
            msg = "Firestore client not available."
            raise RuntimeError(msg)

        docs = await self._execute_blocking(_blocking_op)
        if docs:
            self.logger.info(f"Retrieved {len(docs)} documents from {collection_name}.")
            return docs
        return []

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="firestore_delete_document",
    )
    async def delete_document(
        self,
        collection_name: str,
        doc_id: str,
        is_public: bool = False,
    ) -> bool:
        """Deletes a document by its ID."""
        if not self._firestore_enabled:
            self.logger.debug(
                f"Firestore disabled. Skipping delete_document for {collection_name}/{doc_id}.",
            )
            return False

        collection_path = self._get_collection_path(collection_name, is_public)
        if not collection_path:
            return False

        def _blocking_op():
            if self._db:
                self._db.collection(collection_path).document(str(doc_id)).delete()
            else:
                msg = "Firestore client not available."
                raise RuntimeError(msg)

        result = await self._execute_blocking(_blocking_op)
        if result is not None:
            self.logger.info(f"Document {doc_id} deleted from {collection_name}.")
            return True
        return False


# Global database instance.
# It should be initialized asynchronously in the main application entry point.
db_manager = FirestoreManager()
