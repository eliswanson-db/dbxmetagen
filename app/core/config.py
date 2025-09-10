"""
Core configuration and state management module.
Handles YAML config loading, session state, and Databricks client setup.
"""

import time
import streamlit as st
import yaml
import os
import logging
from typing import Dict, Any, Optional, List
from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages application configuration and state."""

    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize all session state variables in one place"""
        defaults = {
            "config": {},
            "job_runs": {},
            "workspace_client": None,
            "auto_refresh": False,
            "refresh_interval": 30,
            "selected_tables": [],
            "current_metadata": None,
            "job_creation_status": "idle",
            "performance_metrics": {},
        }

        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def load_default_config(self) -> Dict[str, Any]:
        """Load configuration from cached or variables.yml file."""
        # Check for cached config
        cached_config = self._get_cached_config()
        if cached_config is not None:
            logger.info("Using cached configuration")
            return cached_config

        # Load variables.yml and set it up as config
        config = self._load_variables_yml()
        return self._cache_and_return_config(config)

    def _load_variables_yml(self) -> Dict[str, Any]:
        """Load configuration from variables.yml file."""
        yaml_path = "./variables.yml"  # Root of project

        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"variables.yml not found at {yaml_path}")

        logger.info(f"Loading variables.yml from {yaml_path}")
        with open(yaml_path, "r") as f:
            raw_config = yaml.safe_load(f)

        if not raw_config or "variables" not in raw_config:
            raise ValueError(
                "Invalid variables.yml structure - missing 'variables' key"
            )

        # Extract default values from variables.yml structure
        config = {}
        variables = raw_config["variables"]

        for key, value_config in variables.items():
            if isinstance(value_config, dict) and "default" in value_config:
                config[key] = value_config["default"]
            else:
                config[key] = value_config

        logger.info(f"Successfully loaded variables.yml with {len(config)} keys")
        return config

    def _get_cached_config(self) -> Optional[Dict[str, Any]]:
        """Check if we have a valid cached configuration and return it."""
        if st.session_state.config and isinstance(st.session_state.config, dict):
            return st.session_state.config
        return None

    def _apply_host_override(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Override host from environment if available."""
        env_host = os.environ.get("DATABRICKS_HOST")
        if env_host:
            config["host"] = env_host
            logger.info("Using DATABRICKS_HOST environment variable")
        return config

    def _cache_and_return_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Cache configuration in session state and return."""
        config = self._apply_host_override(config)
        st.session_state.config = config
        logger.info(f"Configuration loaded with {len(config)} keys")
        return config


class DatabricksClientManager:
    """Manages Databricks workspace client setup and authentication."""

    # Class constants for better maintainability
    REQUIRED_ENV_VAR = "DATABRICKS_HOST"
    TOKEN_ENV_VARS = ["DATABRICKS_TOKEN", "DATABRICKS_ACCESS_TOKEN"]
    FORWARDED_TOKEN_HEADERS = ["x-forwarded-access-token", "x-forwarded-user-token"]
    QUERY_TOKEN_PARAMS = ["token", "access_token"]

    @staticmethod
    def setup_client() -> bool:
        """
        Initialize Databricks workspace client with user impersonation.

        Returns:
            bool: True if setup successful, False otherwise
        """
        # Skip if already initialized
        if DatabricksClientManager._is_client_already_initialized():
            logger.info("Workspace client already initialized")
            return True

        try:
            # Step 1: Validate environment
            host = DatabricksClientManager._validate_environment()
            if not host:
                return False

            # Step 2: Create authenticated client
            client = DatabricksClientManager._create_authenticated_client(host)
            if not client:
                return False

            # Step 3: Test connection
            user_info = DatabricksClientManager._test_client_connection(client)
            if not user_info:
                return False

            # Step 4: Store successful client
            DatabricksClientManager._store_client_in_session(client, user_info)
            return True

        except Exception as e:
            DatabricksClientManager._handle_setup_error(e)
            return False

    @staticmethod
    def _is_client_already_initialized() -> bool:
        """Check if workspace client is already initialized."""
        return st.session_state.get("workspace_client") is not None

    @staticmethod
    def _validate_environment() -> Optional[str]:
        """
        Validate required environment variables.

        Returns:
            str: Databricks host URL if valid, None otherwise
        """
        host = os.environ.get(DatabricksClientManager.REQUIRED_ENV_VAR)

        if not host:
            error_msg = f"❌ {DatabricksClientManager.REQUIRED_ENV_VAR} environment variable not set"
            logger.error(error_msg)
            st.error(error_msg)
            return None

        # Normalize host URL by adding protocol if missing
        normalized_host = DatabricksClientManager._normalize_host_url(host)

        if not DatabricksClientManager._is_valid_host_url(normalized_host):
            error_msg = f"❌ Invalid host URL format: {host}"
            logger.error(error_msg)
            st.error(error_msg)
            return None

        logger.info(f"Using Databricks host: {normalized_host}")
        return normalized_host

    @staticmethod
    def _normalize_host_url(host: str) -> str:
        """
        Normalize host URL by adding https:// protocol if missing.

        Args:
            host: Raw host URL that may or may not include protocol

        Returns:
            str: Normalized host URL with protocol
        """
        host = host.strip()
        if not host.startswith(("https://", "http://")):
            host = f"https://{host}"
        return host

    @staticmethod
    def _is_valid_host_url(host: str) -> bool:
        """Validate host URL format."""
        return (
            host.startswith(("https://", "http://"))
            and len(host.strip()) > 10
            and "." in host
        )

    @staticmethod
    def _create_authenticated_client(host: str) -> Optional[WorkspaceClient]:
        """
        Create Databricks client with best available authentication.

        Args:
            host: Databricks workspace host URL

        Returns:
            WorkspaceClient: Authenticated client or None if failed
        """
        try:
            # Try user token authentication first
            user_token = DatabricksClientManager._extract_user_token()

            if user_token:
                logger.info("Creating client with user token authentication")
                return WorkspaceClient(host=host, token=user_token)
            else:
                logger.info("Creating client with default authentication")
                return WorkspaceClient(host=host)

        except Exception as e:
            logger.error(f"Failed to create Databricks client: {str(e)}")
            return None

    @staticmethod
    def _test_client_connection(client: WorkspaceClient) -> Optional[Dict[str, str]]:
        """
        Test client connection and get user information.

        Args:
            client: Databricks workspace client

        Returns:
            dict: User information if successful, None if failed
        """
        try:
            logger.info("Testing Databricks client connection...")
            current_user = client.current_user.me()

            user_info = {
                "user_name": current_user.user_name,
                "display_name": getattr(
                    current_user, "display_name", current_user.user_name
                ),
                "user_id": getattr(current_user, "id", "unknown"),
            }

            logger.info(f"Successfully connected as: {user_info['user_name']}")
            return user_info

        except Exception as e:
            error_msg = f"Failed to authenticate with Databricks: {str(e)}"
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
            return None

    @staticmethod
    def _store_client_in_session(client: WorkspaceClient, user_info: Dict[str, str]):
        """
        Store successful client and user info in session state.

        Args:
            client: Authenticated Databricks client
            user_info: User information dictionary
        """
        st.session_state.workspace_client = client
        st.session_state.databricks_user_info = user_info

        logger.info(f"Stored client in session for user: {user_info['user_name']}")

        # Optional: Show success message to user
        st.success(f"✅ Connected to Databricks as {user_info['display_name']}")

    @staticmethod
    def _handle_setup_error(error: Exception):
        """Handle setup errors with proper logging and user feedback."""
        error_msg = f"Failed to setup Databricks client: {str(error)}"
        logger.error(error_msg, exc_info=True)
        st.error(f"❌ Connection Error: {str(error)}")

    @staticmethod
    def _extract_user_token() -> Optional[str]:
        """
        Extract user token from various sources with improved error handling.

        Returns:
            str: User token if found, None otherwise
        """
        token_sources = [
            DatabricksClientManager._try_streamlit_context_token,
            DatabricksClientManager._try_query_params_token,
            DatabricksClientManager._try_environment_token,
        ]

        for source_func in token_sources:
            try:
                token = source_func()
                if token:
                    logger.debug(f"Token found via {source_func.__name__}")
                    return token
            except Exception as e:
                logger.debug(
                    f"Token extraction failed for {source_func.__name__}: {str(e)}"
                )
                continue

        logger.info("No user token found, will use default authentication")
        return None

    @staticmethod
    def _try_streamlit_context_token() -> Optional[str]:
        """Extract token from Streamlit request context."""
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        if not ctx or not hasattr(ctx, "session_info"):
            return None

        session_info = ctx.session_info
        if not hasattr(session_info, "headers"):
            return None

        headers = session_info.headers

        # Try forwarded token headers
        for header_name in DatabricksClientManager.FORWARDED_TOKEN_HEADERS:
            token = headers.get(header_name)
            if token:
                return token

        # Try authorization header
        auth_header = headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header.replace("Bearer ", "")

        return None

    @staticmethod
    def _try_query_params_token() -> Optional[str]:
        """Extract token from query parameters."""
        try:
            query_params = st.query_params
            for param_name in DatabricksClientManager.QUERY_TOKEN_PARAMS:
                token = query_params.get(param_name)
                if token:
                    return token
        except Exception:
            pass
        return None

    @staticmethod
    def _try_environment_token() -> Optional[str]:
        """Extract token from environment variables."""
        for env_var in DatabricksClientManager.TOKEN_ENV_VARS:
            token = os.environ.get(env_var)
            if token:
                return token
        return None

    @staticmethod
    def get_current_user_info() -> Optional[Dict[str, str]]:
        """
        Get current user information from session state.

        Returns:
            dict: User information or None if not available
        """
        return st.session_state.get("databricks_user_info")

    @staticmethod
    def is_client_ready() -> bool:
        """Check if client is ready for use."""
        return st.session_state.get("workspace_client") is not None

    @staticmethod
    def reset_client():
        """Reset/clear the current client (useful for testing or re-authentication)."""
        if "workspace_client" in st.session_state:
            del st.session_state.workspace_client
        if "databricks_user_info" in st.session_state:
            del st.session_state.databricks_user_info
        logger.info("Databricks client reset")
