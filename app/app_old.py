# from app.test_job_manager import current_user
from utils import get_current_user_email
import streamlit as st
import pandas as pd
import yaml
import json
import time
import os
import io
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (
    JobSettings,
    NotebookTask,
    JobCluster,
    ClusterSpec,
    JobEnvironment,
    Task,
    PerformanceTarget,
    # Environment,
    # EnvironmentSpec,
)
from databricks.sdk.service.compute import Environment
import base64
import logging
import sys

# Configure logging for Databricks Apps (writes to stdout/stderr)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # This goes to stdout for Databricks Apps
        logging.StreamHandler(sys.stderr),  # This goes to stderr for Databricks Apps
    ],
)
logger = logging.getLogger(__name__)

# Log app startup
logger.info("🚀 DBX MetaGen App starting up...")
print(
    "🚀 [STARTUP] DBX MetaGen App starting up..."
)  # Also use print for immediate visibility

# Import job manager with error handling
try:
    from job_manager import DBXMetaGenJobManager

    JOB_MANAGER_AVAILABLE = True
    JOB_MANAGER_IMPORT_ERROR = None
    print("[IMPORT] ✅ Successfully imported DBXMetaGenJobManager")
except ImportError as e:
    JOB_MANAGER_AVAILABLE = False
    JOB_MANAGER_IMPORT_ERROR = str(e)
    print(f"[IMPORT] ❌ Failed to import DBXMetaGenJobManager: {e}")
except Exception as e:
    JOB_MANAGER_AVAILABLE = False
    JOB_MANAGER_IMPORT_ERROR = str(e)
    print(f"[IMPORT] ❌ Unexpected error importing DBXMetaGenJobManager: {e}")

# Configure page
st.set_page_config(
    page_title="DBX MetaGen",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state():
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


# Initialize session state
initialize_session_state()

# Debug mode configuration
# Configure logging
DEBUG_MODE = os.environ.get("DEBUG", "false").lower() == "true"
LOG_LEVEL = logging.DEBUG if DEBUG_MODE else logging.INFO

# Set up logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "[%(levelname)s] %(asctime)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(LOG_LEVEL)

# Initialize debug log storage for Streamlit
if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = []

# Log debug mode status at startup
if DEBUG_MODE:
    logger.debug("Debug mode enabled")
    st.session_state.debug_logs.append(
        f"[DEBUG] {datetime.now().strftime('%H:%M:%S')} - Debug mode enabled"
    )


def debug_log(message: str):
    """Log debug messages using proper logging and show in Streamlit"""
    logger.debug(message)

    # Always store debug messages in session state, but only display if DEBUG_MODE is on
    timestamp = datetime.now().strftime("%H:%M:%S")
    debug_entry = f"[DEBUG] {timestamp} - {message}"

    # Ensure debug_logs exists in session state
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []

    st.session_state.debug_logs.append(debug_entry)

    # Keep only last 50 debug messages to avoid memory issues
    if len(st.session_state.debug_logs) > 50:
        st.session_state.debug_logs = st.session_state.debug_logs[-50:]


def handle_databricks_error(error: Exception, operation: str = "operation"):
    """Provide specific error handling and user guidance for Databricks operations"""
    error_msg = str(error).lower()

    if "authentication" in error_msg or "unauthorized" in error_msg:
        st.error(
            "🔐 Authentication failed. Please check your Databricks configuration."
        )
        with st.expander("💡 How to fix authentication issues"):
            st.markdown(
                """
            1. **Check environment variables**: Ensure `DATABRICKS_HOST` is set correctly
            2. **Verify service principal**: Ensure the app service principal has proper permissions
            3. **Token issues**: If using tokens, verify they haven't expired
            4. **Workspace access**: Ensure the service principal is added to the workspace
            """
            )
    elif "permission" in error_msg or "forbidden" in error_msg:
        st.error("🚫 Insufficient permissions for this operation.")
        with st.expander("💡 How to fix permission issues"):
            st.markdown(
                """
            1. **Contact workspace admin** for the required permissions
            2. **Check service principal roles** in the workspace
            3. **Verify catalog/schema access** if working with data
            4. **Job creation permissions** may require workspace admin role
            """
            )
    elif "cluster" in error_msg:
        st.error("🖥️ Cluster configuration issue.")
        with st.expander("💡 How to fix cluster issues"):
            st.markdown(
                """
            1. **Try a smaller cluster size** (reduce workers)
            2. **Check node type availability** in your workspace region
            3. **Verify cluster permissions** for job creation
            4. **Check workspace quotas** for cluster creation
            """
            )
    elif "network" in error_msg or "connection" in error_msg:
        st.error("🌐 Network connectivity issue.")
        with st.expander("💡 How to fix network issues"):
            st.markdown(
                """
            1. **Check internet connection**
            2. **Verify workspace URL** is correct and accessible
            3. **Firewall rules** may be blocking access
            4. **Try again in a few minutes** - temporary network issues
            """
            )
    else:
        st.error(f"❌ {operation.title()} failed: {str(error)}")
        with st.expander("💡 Troubleshooting steps"):
            st.markdown(
                """
            1. **Check the error details** above for specific issues
            2. **Try the operation again** - it may be a temporary issue
            3. **Check Databricks workspace status** for any outages
            4. **Contact support** if the issue persists
            """
            )

    # Always provide fallback action
    st.info(
        "💡 You can also perform this operation manually in the Databricks workspace."
    )


def validate_table_names(table_names: List[str]) -> tuple[List[str], List[str]]:
    """Validate table names and return valid/invalid lists with detailed feedback"""
    valid_tables = []
    invalid_tables = []

    for table in table_names:
        table = table.strip()
        if not table:
            continue

        # Check format: catalog.schema.table
        parts = table.split(".")
        if len(parts) != 3:
            invalid_tables.append(
                f"{table} (invalid format - need catalog.schema.table)"
            )
            continue

        catalog, schema, table_name = parts

        # Check for empty parts
        if not all(part.strip() for part in parts):
            invalid_tables.append(f"{table} (contains empty parts)")
            continue

        # Check for valid characters (alphanumeric, underscore, hyphen)
        if not all(part.replace("_", "").replace("-", "").isalnum() for part in parts):
            invalid_tables.append(
                f"{table} (invalid characters - use only letters, numbers, _, -)"
            )
            continue

        # Check length (reasonable limits)
        if any(len(part) > 100 for part in parts):
            invalid_tables.append(f"{table} (part too long - max 100 characters)")
            continue

        valid_tables.append(table)

    return valid_tables, invalid_tables


def display_table_validation_results(
    valid_tables: List[str], invalid_tables: List[str]
):
    """Display validation results with user-friendly feedback"""
    if valid_tables:
        st.success(f"✅ {len(valid_tables)} valid table names found")
        if len(valid_tables) <= 10:
            with st.expander("Valid Tables"):
                for table in valid_tables:
                    st.write(f"• {table}")
        else:
            with st.expander(f"Valid Tables (showing first 10 of {len(valid_tables)})"):
                for table in valid_tables[:10]:
                    st.write(f"• {table}")
                st.write(f"... and {len(valid_tables) - 10} more")

    if invalid_tables:
        st.error(f"❌ {len(invalid_tables)} invalid table names found")
        with st.expander("Invalid Tables - Click to see issues"):
            for table_error in invalid_tables:
                st.write(f"• {table_error}")

        with st.expander("💡 Table Name Requirements"):
            st.markdown(
                """
            **Valid format**: `catalog.schema.table`
            
            **Rules**:
            - Must have exactly 3 parts separated by dots
            - Each part can contain letters, numbers, underscores (_), and hyphens (-)
            - No empty parts allowed
            - Maximum 100 characters per part
            
            **Examples**:
            - ✅ `my_catalog.default.users`
            - ✅ `analytics.sales_data.transactions`
            - ❌ `catalog.table` (missing schema)
            - ❌ `catalog..table` (empty schema)
            - ❌ `catalog.schema.table with spaces` (contains spaces)
            """
            )


class DBXMetaGenApp:
    def __init__(self):
        self.setup_client()

    def setup_client(self):
        """Initialize Databricks workspace client with user impersonation"""
        # Check if already initialized
        if st.session_state.get("workspace_client") is not None:
            return

        try:
            host = os.environ.get("DATABRICKS_HOST")
            if host:
                # Try to get user token from Streamlit context for impersonation
                user_token = None
                try:
                    # Method 1: Check Streamlit's request context
                    from streamlit.runtime.scriptrunner import get_script_run_ctx

                    ctx = get_script_run_ctx()
                    if ctx and hasattr(ctx, "session_info"):
                        session_info = ctx.session_info
                        if hasattr(session_info, "headers"):
                            headers = session_info.headers
                            user_token = (
                                headers.get("x-forwarded-access-token")
                                or headers.get("x-forwarded-user-token")
                                or headers.get("authorization", "").replace(
                                    "Bearer ", ""
                                )
                            )
                except Exception as e:
                    logger.debug(f"Failed to extract user token from headers: {str(e)}")
                    # Continue to try other token extraction methods

                # Method 2: Try query parameters (sometimes user token is passed this way)
                if not user_token:
                    try:
                        query_params = st.query_params
                        user_token = query_params.get("token") or query_params.get(
                            "access_token"
                        )
                    except Exception as e:
                        logger.debug(
                            f"Failed to extract user token from query params: {str(e)}"
                        )
                        # Continue to try environment variable method

                # Method 3: Environment variable (if set by app framework)
                if not user_token:
                    user_token = os.environ.get("DATABRICKS_USER_TOKEN")

                if user_token:
                    # Create user-impersonated client
                    workspace_client = WorkspaceClient(host=host, token=user_token)
                    st.session_state.user_impersonated = True

                    # Get the actual user to confirm impersonation
                    try:
                        current_user = workspace_client.current_user.me()
                        st.success(
                            f"✅ Connected with user impersonation as: {current_user.user_name}"
                        )
                        logger.debug(
                            f"User impersonation successful for: {current_user.user_name}"
                        )
                    except Exception:
                        st.success("✅ Connected with user impersonation")
                        logger.debug("Workspace client initialized with user token")
                else:
                    workspace_client = WorkspaceClient(host=host)
                    st.session_state.user_impersonated = False
                    st.info(
                        "ℹ️ Connected with service principal (user token not available)"
                    )
                    logger.debug("Workspace client initialized with service principal")

                st.session_state.workspace_client = workspace_client
                st.session_state.databricks_host = host
            else:
                st.warning(
                    "⚠️ DATABRICKS_HOST environment variable not found. Some features may not work."
                )
        except Exception as e:
            st.error(f"❌ Failed to initialize Databricks client: {str(e)}")
            st.session_state.workspace_client = None
            st.session_state.databricks_host = None

    def load_default_config(self) -> Dict[str, Any]:
        """
        Load default configuration from variables.yml with caching.

        Returns:
            Dictionary containing configuration values
        """
        # Check cache first
        cached_config = self._get_cached_config()
        if cached_config is not None:
            return cached_config

        # Try to load from files
        loaded_config = self._load_config_from_available_paths()

        # Cache and return the result
        return self._cache_and_return_config(loaded_config)

    def _get_cached_config(self) -> Optional[Dict[str, Any]]:
        """
        Check if configuration is already loaded in session state.

        Returns:
            Cached configuration or None if not cached
        """
        return st.session_state.get("default_config_loaded")

    def _load_config_from_available_paths(self) -> Optional[Dict[str, Any]]:
        """
        Try loading configuration from multiple possible paths.

        Returns:
            Loaded configuration dictionary or None if not found
        """
        config_paths = self._get_config_search_paths()

        for config_path in config_paths:
            config = self._try_load_single_config_file(config_path)
            if config is not None:
                st.info(f"✅ Configuration loaded from {config_path}")
                return config

        return None

    def _get_config_search_paths(self) -> List[str]:
        """
        Get list of paths to search for configuration files.

        Returns:
            List of file paths to try
        """
        return ["variables.yml", "../variables.yml", "../../variables.yml"]

    def _try_load_single_config_file(
        self, config_path: str
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt to load and process a single configuration file.

        Args:
            config_path: Path to configuration file

        Returns:
            Processed configuration dictionary or None if failed
        """
        try:
            raw_config = self._load_yaml_file(config_path)
            if raw_config is None:
                return None

            return self._process_raw_config(raw_config)

        except FileNotFoundError:
            return None  # Try next path
        except Exception as e:
            st.warning(f"⚠️ Error loading {config_path}: {str(e)}")
            return None

    def _load_yaml_file(self, config_path: str) -> Optional[Dict[str, Any]]:
        """
        Load and parse YAML file.

        Args:
            config_path: Path to YAML file

        Returns:
            Parsed YAML content or None if invalid format
        """
        with open(config_path, "r", encoding="utf-8") as f:
            variables = yaml.safe_load(f)

        if "variables" not in variables:
            st.warning(f"⚠️ {config_path} found but has unexpected format")
            return None

        return variables

    def _process_raw_config(self, raw_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw configuration by extracting defaults and applying post-processing.

        Args:
            raw_config: Raw configuration from YAML file

        Returns:
            Processed configuration dictionary
        """
        # Extract default values
        default_config = self._extract_default_values(raw_config)

        # Apply post-processing
        default_config = self._apply_config_post_processing(default_config)

        return default_config

    def _extract_default_values(self, raw_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract default values from variables.yml structure.

        Args:
            raw_config: Raw configuration dictionary

        Returns:
            Dictionary with default values extracted
        """
        default_config = {}

        for key, config in raw_config["variables"].items():
            default_config[key] = config.get("default")

        return default_config

    def _apply_config_post_processing(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply post-processing to configuration (host override, sanitization).

        Args:
            config: Configuration dictionary to process

        Returns:
            Post-processed configuration dictionary
        """
        # Handle host override from session state
        config = self._apply_host_override(config)

        # Sanitize unresolved placeholders
        config = self._sanitize_placeholder_values(config)

        return config

    def _apply_host_override(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override host value from session state if available.

        Args:
            config: Configuration dictionary

        Returns:
            Configuration with host potentially overridden
        """
        if "host" in config and st.session_state.get("databricks_host"):
            config["host"] = st.session_state.databricks_host

        return config

    def _sanitize_placeholder_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize unresolved placeholder values in configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Configuration with placeholders sanitized
        """
        # Sanitize owner_user placeholder
        owner_user_val = (config.get("owner_user") or "").strip()
        if self._is_unresolved_placeholder(owner_user_val):
            config["owner_user"] = ""

        return config

    def _is_unresolved_placeholder(self, value: str) -> bool:
        """
        Check if a value is an unresolved placeholder.

        Args:
            value: String value to check

        Returns:
            True if value is an unresolved placeholder
        """
        return value.startswith("${") and value.endswith("}")

    def _cache_and_return_config(
        self, config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Cache configuration in session state and return it.

        Args:
            config: Configuration to cache, or None to use built-in defaults

        Returns:
            Final configuration dictionary
        """
        if config is not None:
            st.session_state.default_config_loaded = config
            return config

        # Fallback to built-in defaults
        default_config = self.get_builtin_defaults()
        st.session_state.default_config_loaded = default_config
        st.info("📋 No variables.yml found, using built-in defaults")
        return default_config

    def get_builtin_defaults(self) -> Dict[str, Any]:
        """Get built-in default configuration as fallback"""
        # Get current host from environment if available
        current_host = st.session_state.get("databricks_host") or os.environ.get(
            "DATABRICKS_HOST", ""
        )

        default_config = {
            "catalog_name": "dbxmetagen",
            "host": current_host,
            "allow_data": True,
            "sample_size": 5,
            "mode": "comment",
            "disable_medical_information_value": True,
            "allow_data_in_comments": True,
            "add_metadata": True,
            "include_deterministic_pi": True,
            "apply_ddl": False,
            "ddl_output_format": "sql",
            "reviewable_output_format": "tsv",
            "model": "databricks-claude-3-7-sonnet",
            "max_tokens": 4096,
            "temperature": 0.1,
            "columns_per_call": 5,
            "schema_name": "metadata_results",
            "volume_name": "generated_metadata",
            "owner_user": "",
        }
        return default_config

    def render_sidebar_config(self):
        """Render configuration sidebar"""
        st.sidebar.header("⚙️ Configuration")

        # Load default config if not set
        # if not st.session_state.config:
        #    st.session_state.config = self.load_default_config()
        

        config = st.session_state.config

        # Core Settings
        with st.sidebar.expander("🏗️ Core Settings", expanded=True):
            config["catalog_name"] = st.text_input(
                "Catalog Name",
                value=config.get("catalog_name", ""),
                help="Target catalog where data, models, and files are stored",
            )

            config["host"] = st.text_input(
                "Databricks Host",
                value=config.get("host", ""),
                help="Your Databricks workspace URL",
            )

            config["schema_name"] = st.text_input(
                "Schema Name",
                value=config.get("schema_name", ""),
                help="Schema where results will be stored",
            )

            # New: Owner user email to resolve workspace path for notebooks
            owner_default = config.get("owner_user", "") or ""
            if isinstance(owner_default, str) and owner_default.startswith("${"):
                owner_default = ""
            config["owner_user"] = st.text_input(
                "Owner User Email",
                value=owner_default,
                help=(
                    "User home under which the bundle was deployed (email). "
                    "Used to resolve the notebook path so jobs run against your user folder."
                ),
            )

        # Data Settings
        with st.sidebar.expander("📊 Data Settings"):
            config["allow_data"] = st.checkbox(
                "Allow Data in Processing",
                value=config.get("allow_data", False),
                help="Set to false to prevent data from being sent to LLMs",
            )

            config["sample_size"] = st.number_input(
                "Sample Size",
                min_value=0,
                max_value=50,
                value=config.get("sample_size", 0),
                help="Number of data samples per column for analysis",
            )

            mode_options = ["comment", "pi", "both"]
            current_mode = config.get("mode", "comment")
            mode_index = (
                mode_options.index(current_mode) if current_mode in mode_options else 0
            )
            config["mode"] = st.selectbox(
                "Processing Mode",
                options=mode_options,
                index=mode_index,
                help="Mode of operation: generate comments, identify PII, or both",
            )

        # Advanced Settings
        with st.sidebar.expander("🔧 Advanced Settings"):
            model_options = [
                "databricks-claude-3-7-sonnet",
                "databricks-meta-llama-3-3-70b-instruct",
                "databricks-claude-3-5-sonnet",
            ]
            current_model = config.get("model", "")
            model_index = (
                model_options.index(current_model)
                if current_model in model_options
                else 0
            )
            config["model"] = st.selectbox(
                "LLM Model",
                options=model_options,
                index=model_index,
                help="LLM model for metadata generation",
            )

            config["apply_ddl"] = st.checkbox(
                "Apply DDL Directly",
                value=config.get("apply_ddl", False),
                help="⚠️ WARNING: This will modify your tables directly",
            )

            config["temperature"] = st.slider(
                "Model Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(config.get("temperature", 0.1)),
                step=0.1,
                help="Model creativity level (0.0 = deterministic, 1.0 = creative)",
            )

        st.session_state.config = config

        # Save/Load Config
        st.sidebar.markdown("---")
        col1, col2 = st.sidebar.columns(2)

        with col1:
            if st.button("💾 Save Config"):
                self.save_config_to_file()

        with col2:
            uploaded_config = st.file_uploader(
                "📁 Load Config", type=["yml", "yaml"], key="config_upload"
            )
            if uploaded_config:
                self.load_config_from_file(uploaded_config)

    def save_config_to_file(self):
        """Save current configuration to downloadable file"""
        config_yaml = yaml.dump(
            {
                "variables": {
                    k: {"default": v} for k, v in st.session_state.config.items()
                }
            },
            default_flow_style=False,
        )

        st.sidebar.download_button(
            label="⬇️ Download Config",
            data=config_yaml,
            file_name=f"dbxmetagen_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yml",
            mime="text/yaml",
        )

    def load_config_from_file(self, uploaded_file):
        """Load configuration from uploaded file"""
        try:
            config_data = yaml.safe_load(uploaded_file)
            if "variables" in config_data:
                new_config = {
                    k: v.get("default", v) for k, v in config_data["variables"].items()
                }
                st.session_state.config.update(new_config)
                st.sidebar.success("✅ Configuration loaded successfully!")
                st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Error loading config: {str(e)}")

    def render_unified_table_management(self):
        """Render unified table management and job creation section - orchestrates the entire table management UI"""
        st.header("📋 Table Management & Job Creation")

        # Split into focused sub-methods
        tables = self._render_table_input_section()
        if tables:
            self._render_table_action_buttons(tables)
            self._handle_job_dialog_display()
        else:
            self._show_no_tables_warning()

    def _render_table_input_section(self) -> List[str]:
        """Render table input section with manual entry and CSV upload tabs"""
        st.subheader("📝 Table Selection")

        tab1, tab2 = st.tabs(["📝 Manual Entry", "📤 Upload CSV"])

        with tab1:
            table_names_input = st.text_area(
                "Table Names (one per line)",
                height=200,
                placeholder="catalog.schema.table1\ncatalog.schema.table2\n...",
                help="Enter fully qualified table names, one per line",
                value="\n".join(st.session_state.get("selected_tables", [])),
                key="unified_table_input",
            )

        with tab2:
            uploaded_file = st.file_uploader(
                "Upload table_names.csv",
                type=["csv"],
                help="CSV file with 'table_name' column containing fully qualified table names",
                key="unified_csv_upload",
            )

            if uploaded_file:
                tables_from_csv = self.process_uploaded_csv(uploaded_file)
                if tables_from_csv:
                    st.session_state.selected_tables = tables_from_csv
                    st.rerun()

        return self._parse_and_store_tables(table_names_input)

    def _parse_and_store_tables(self, table_names_input: str) -> List[str]:
        """Parse table input and store in session state"""
        if table_names_input.strip():
            tables = [t.strip() for t in table_names_input.split("\n") if t.strip()]
            st.session_state.selected_tables = tables
            return tables
        else:
            return st.session_state.get("selected_tables", [])

    def _render_table_action_buttons(self, tables: List[str]):
        """Render action buttons for table operations"""
        st.info(f"📊 {len(tables)} table(s) selected")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("🔍 Validate Tables", type="secondary"):
                self.validate_tables(tables)

        with col2:
            if st.button("💾 Download CSV"):
                self.save_table_list(tables)

        with col3:
            self._render_job_creation_button(tables)

        with col4:
            if st.button("🗑️ Clear Tables"):
                st.session_state.selected_tables = []
                st.rerun()

    def _render_job_creation_button(self, tables: List[str]):
        """Render job creation button with debug functionality"""
        # TODO: Replace with proper job creation dialog when working
        if st.button("🔧 Create and Run Job", help="Debug job manager"):
            self._debug_job_manager_creation(tables)

    def _debug_job_manager_creation(self, tables: List[str]):
        """Debug job manager creation - temporary method until proper job creation works"""
        st.write("**🔧 Running Job Manager...**")

        if not JOB_MANAGER_AVAILABLE:
            st.error("❌ Job Manager not available!")
            st.info("💡 Check import errors and restart the app")
            st.write(f"Import error: {JOB_MANAGER_IMPORT_ERROR}")
            logger.error(f"Job manager debug failed: {JOB_MANAGER_IMPORT_ERROR}")
            return

        try:
            job_manager = DBXMetaGenJobManager(st.session_state.workspace_client)
            current_user_email = get_current_user_email()
            from datetime import datetime

            job_name = f"dbxmetagen_debug_job_{current_user_email}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            st.write(f"• 🚀 Creating job: {job_name}")
            logger.info(f"Debug job creation starting: {job_name}")

            # 🔧 FIX: Actually capture the return values!
            job_id, run_id = job_manager.create_metadata_job(
                job_name=job_name,
                tables=tables,
                cluster_size="Small (1-2 workers)",
                config=st.session_state.config,
                user_email=current_user_email,
            )

            st.write(f"• ✅ Job created with ID: {job_id}")
            st.write(f"• ✅ Run started with ID: {run_id}")
            logger.info(
                f"Debug job created successfully - job_id: {job_id}, run_id: {run_id}"
            )

            # 🔧 FIX: Store job run info in session state like the working method does
            self._store_debug_job_run_info(job_id, run_id, job_name, tables)

            # Diagnostics
            st.success("✅ Job created and stored for tracking!")
            st.write(f"- Job Manager type: {type(job_manager)}")
            st.write(f"- Job ID: {job_id}, Run ID: {run_id}")
            st.write(
                f"- Session state job runs count: {len(st.session_state.get('job_runs', {}))}"
            )

        except Exception as test_error:
            st.error(f"❌ Job creation failed: {str(test_error)}")
            logger.error(f"Debug job creation failed: {str(test_error)}", exc_info=True)
            with st.expander("Error Details"):
                import traceback

                st.code(traceback.format_exc())

    def _show_no_tables_warning(self):
        """Show warning when no tables are selected"""
        st.warning(
            "⚠️ No tables selected. Please enter table names or upload a CSV file."
        )
        st.info(
            "💡 Use the tabs above to manually enter table names or upload a CSV file"
        )

    def _store_debug_job_run_info(
        self, job_id: int, run_id: int, job_name: str, tables: List[str]
    ):
        """Store debug job run information in session state for tracking"""
        from datetime import datetime

        # Initialize job_runs if it doesn't exist
        if "job_runs" not in st.session_state:
            st.session_state.job_runs = {}
            logger.debug("Initialized empty job_runs in session state")

        # Store job run info using same format as working method
        st.session_state.job_runs[run_id] = {
            "job_id": job_id,
            "job_name": job_name,
            "run_id": run_id,
            "tables": tables,
            "status": "PENDING",
            "start_time": datetime.now(),
            "mode": st.session_state.config.get("mode", "comment"),
            "cluster_size": "Small (1-2 workers)",
            "table_count": len(tables),
        }

        logger.info(
            f"Stored debug job run in session state: run_id={run_id}, job_id={job_id}"
        )
        logger.debug(
            f"Total job runs in session state: {len(st.session_state.job_runs)}"
        )

        # Show confirmation
        st.info(f"📝 Job run {run_id} stored in session state for tracking")

    def _handle_job_dialog_display(self):
        """Handle job dialog display logic"""
        if st.session_state.get("show_job_dialog", False):
            self.show_job_creation_dialog(st.session_state.get("job_dialog_tables", []))
            st.session_state.show_job_dialog = False

    def show_job_creation_dialog(self, tables: List[str]):
        """Show job creation dialog with configuration options"""
        print(f"[UI] show_job_creation_dialog called with {len(tables)} tables")
        debug_log(f"show_job_creation_dialog called with {len(tables)} tables")

        st.markdown("---")
        st.subheader("🚀 Create Metadata Generation Job")

        with st.container():
            col1, col2 = st.columns([2, 1])

            with col1:
                job_name = st.text_input(
                    "Job Name",
                    value=f"dbxmetagen_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    help="Name for the metadata generation job",
                )

                cluster_size = st.selectbox(
                    "Cluster Size",
                    options=[
                        "Small (1-2 workers)",
                        "Medium (2-4 workers)",
                        "Large (4-8 workers)",
                    ],
                    index=1,
                    help="Cluster size for the job",
                )

                # Show tables that will be processed
                with st.expander(f"📋 Tables to Process ({len(tables)})"):
                    for i, table in enumerate(tables[:10], 1):
                        st.write(f"{i}. {table}")
                    if len(tables) > 10:
                        st.write(f"... and {len(tables) - 10} more tables")

            with col2:
                st.markdown("**Current Configuration:**")
                config_preview = {
                    "Mode": st.session_state.config.get("mode", "comment"),
                    "Model": st.session_state.config.get(
                        "model", "claude-3-7-sonnet"
                    ).split("-")[-1],
                    "Allow Data": st.session_state.config.get("allow_data", False),
                    "Sample Size": st.session_state.config.get("sample_size", 5),
                    "Apply DDL": st.session_state.config.get("apply_ddl", False),
                }

                for key, value in config_preview.items():
                    st.write(f"**{key}:** {value}")

            # Create and run button
            if st.button("🚀 Create & Run Job", type="primary", key="create_job_main"):
                print(f"[UI] Create & Run Job button clicked")
                print(f"[UI]   - job_name: {job_name}")
                print(f"[UI]   - tables: {len(tables)} tables")
                print(f"[UI]   - cluster_size: {cluster_size}")
                print(
                    f"[UI]   - tables list: {tables[:3]}{'...' if len(tables) > 3 else ''}"
                )

                debug_log(
                    f"Create & Run Job button clicked - job_name: {job_name}, tables: {len(tables)}, cluster_size: {cluster_size}"
                )
                logger.debug(
                    f"Create & Run Job button clicked - job_name: {job_name}, tables: {len(tables)}, cluster_size: {cluster_size}"
                )
                logger.info(
                    f"Create & Run Job button clicked - job_name: {job_name}, tables: {len(tables)}, cluster_size: {cluster_size}"
                )
                try:
                    self.create_and_run_job(job_name, tables, cluster_size)
                except Exception as create_job_e:
                    st.error(f"❌ Failed to create job: {create_job_e}")
                    print(f"[UI] ERROR in create_and_run_job: {create_job_e}")
                    import traceback

                    st.code(traceback.format_exc())

    def create_and_run_job(self, job_name: str, tables: List[str], cluster_size: str):
        """Create and run a metadata generation job - orchestrates the entire process"""
        st.write("**📝 Job Creation Process:**")

        # Step-by-step job creation process
        self._validate_job_prerequisites(tables)
        fresh_config = self._load_fresh_config_from_yaml()
        user_email, username = self._extract_user_information()
        fresh_config = self._fix_config_user_contamination(fresh_config, user_email)
        job_manager = self._initialize_job_manager()
        job_name = self._generate_unique_job_name(username)
        self._cleanup_existing_jobs(job_name)
        tables_str = self._format_tables_for_processing(tables)
        job_id, run_id = self._create_and_start_job(
            job_manager, job_name, tables_str, cluster_size, fresh_config, user_email
        )
        self._store_job_info_and_show_success(
            job_id, run_id, job_name, tables, fresh_config, cluster_size
        )

    def _validate_job_prerequisites(self, tables: List[str]):
        """Validate that all prerequisites for job creation are met"""
        # ⚠️ POTENTIAL ISSUE: These exceptions might be caught by over-eager try/catch blocks upstream
        if not st.session_state.workspace_client:
            raise RuntimeError("Databricks client not initialized")
        if not tables:
            raise ValueError("No tables specified for processing")
        if not JOB_MANAGER_AVAILABLE:
            raise RuntimeError(f"Job manager not available: {JOB_MANAGER_IMPORT_ERROR}")

    def _load_fresh_config_from_yaml(self) -> Dict[str, Any]:
        """Load fresh configuration from YAML file to avoid session state contamination"""
        st.write("• 📁 Loading fresh config from variables.yml...")

        import yaml
        import os

        config_path = "./variables.yml"
        if not os.path.exists(config_path):
            config_path = "app/variables.yml"

        # ⚠️ POTENTIAL ISSUE: No error handling if both paths fail
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _extract_user_information(self) -> tuple[str, str]:
        """Extract user email and username from workspace client"""
        current_user = st.session_state.workspace_client.current_user.me()
        user_email = current_user.user_name
        username = user_email.split("@")[0]
        return user_email, username

    def _fix_config_user_contamination(
        self, fresh_config: Dict[str, Any], user_email: str
    ) -> Dict[str, Any]:
        """Fix service principal contamination in config by overriding user values"""
        st.write(
            "• 🛠️ Fixing config current_user to prevent service principal contamination..."
        )
        st.write(
            f"• 🔍 Before override: current_user = {fresh_config.get('current_user', 'NOT_FOUND')}"
        )

        # Override current_user to prevent service principal contamination
        fresh_config["current_user"] = user_email
        st.write(f"• ✅ After override: current_user = {fresh_config['current_user']}")

        # Fix template variables throughout config
        for key, value in fresh_config.items():
            if isinstance(value, str) and "${workspace.current_user.userName}" in value:
                st.write(f"• 🔧 Fixing template in {key}: {value}")
                fresh_config[key] = value.replace(
                    "${workspace.current_user.userName}", user_email
                )
                st.write(f"• ✅ Fixed {key} = {fresh_config[key]}")

        return fresh_config

    def _initialize_job_manager(self):
        """Initialize the DBXMetaGenJobManager"""
        st.write("• 🔧 Initializing job manager...")
        return DBXMetaGenJobManager(st.session_state.workspace_client)

    def _generate_unique_job_name(self, username: str) -> str:
        """Generate a unique timestamped job name"""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"{username}_app_job_{timestamp}"
        st.write(f"• 📝 Job: {job_name}")
        return job_name

    def _cleanup_existing_jobs(self, job_name: str):
        """Find and delete any existing jobs with the same name"""
        st.write(f"• 🔍 Looking for existing job with name: {job_name}")
        existing_jobs = st.session_state.workspace_client.jobs.list(name=job_name)
        existing_job_id = None

        for job in existing_jobs:
            if job.settings.name == job_name:
                existing_job_id = job.job_id
                st.write(f"• 🔍 Found existing job with ID: {existing_job_id}")
                break

        if existing_job_id:
            st.write(f"• 🗑️ Deleting existing job: {existing_job_id}")
            try:
                st.session_state.workspace_client.jobs.delete(existing_job_id)
                st.write("• ✅ Existing job deleted successfully")
            except Exception as e:
                # ⚠️ POTENTIAL ISSUE: This silently continues on delete failure
                st.write(f"• ⚠️ Warning: Could not delete existing job: {e}")
                st.write("• Continuing with job creation...")
        else:
            st.write("• ✅ No existing job found")

    def _format_tables_for_processing(self, tables: List[str]) -> str:
        """Convert tables list to string format expected by job manager"""
        if isinstance(tables, list):
            tables_str = ",".join(tables)
        else:
            tables_str = str(tables)
        st.write(f"• 📊 Tables: {tables_str}")
        return tables_str

    def _create_and_start_job(
        self,
        job_manager,
        job_name: str,
        tables_str: str,
        cluster_size: str,
        fresh_config: Dict[str, Any],
        user_email: str,
    ) -> tuple[int, int]:
        """Create the job using job manager and start execution"""
        st.write("• 🚀 Creating and running job...")

        # ⚠️ CRITICAL: This is where the actual job creation happens - any failure here is buried
        job_id, run_id = job_manager.create_metadata_job(
            job_name=job_name,
            tables=tables_str,
            cluster_size=cluster_size,
            config=fresh_config,
            user_email=user_email,
        )

        st.write(f"• ✅ Job created with ID: {job_id}")
        st.write(f"• ✅ Run started with ID: {run_id}")
        return job_id, run_id

    def _store_job_info_and_show_success(
        self,
        job_id: int,
        run_id: int,
        job_name: str,
        tables: List[str],
        fresh_config: Dict[str, Any],
        cluster_size: str,
    ):
        """Store job information in session state and display success message"""
        from datetime import datetime

        st.session_state.job_runs[run_id] = {
            "job_id": job_id,
            "job_name": job_name,
            "run_id": run_id,
            "tables": tables,
            "status": "PENDING",
            "start_time": datetime.now(),
            "mode": fresh_config.get("mode", "comment"),
            "cluster_size": cluster_size,
            "table_count": len(tables),
        }

        st.success(
            "🎉 Job created and started! Check the 'Job Status' tab to monitor progress."
        )

    def process_uploaded_csv(self, uploaded_file):
        """Process uploaded CSV file and return table list"""
        try:
            df = pd.read_csv(uploaded_file)

            if "table_name" not in df.columns:
                st.error("❌ CSV must contain a 'table_name' column")
                return None

            tables = df["table_name"].dropna().tolist()

            st.success(f"✅ Loaded {len(tables)} tables from CSV")

            # Display preview
            with st.expander("📋 Table Preview"):
                st.dataframe(df.head(10))

            return tables

        except Exception as e:
            st.error(f"❌ Error processing CSV: {str(e)}")
            return None

    def validate_tables(self, tables: List[str]):
        """Validate that tables exist and are accessible"""
        if not st.session_state.workspace_client:
            st.warning("⚠️ Cannot validate tables: Databricks client not initialized")
            st.info(
                "💡 Initialize your Databricks connection in the Configuration tab first"
            )
            logger.warning("validate_tables called without workspace client")
            return

        # First, validate table name format
        format_valid, format_invalid = validate_table_names(tables)

        if format_invalid:
            st.warning("⚠️ Some table names have format issues:")
            display_table_validation_results(format_valid, format_invalid)

            # Ask user if they want to continue with valid names only
            if format_valid:
                if st.button("🔄 Continue with valid table names only"):
                    tables = format_valid
                else:
                    st.info("Please fix the table name formats above and try again.")
                    return
            else:
                st.error(
                    "No valid table names found. Please fix the formats and try again."
                )
                return

        # Then check if tables exist and are accessible
        with st.spinner("🔍 Checking table existence and access..."):
            accessible_tables = []
            inaccessible_tables = []

            for table in format_valid:
                try:
                    if st.session_state.workspace_client.tables.exists(table):
                        # Also try to get basic info to verify access
                        try:
                            st.session_state.workspace_client.tables.get(table)
                            accessible_tables.append(table)
                        except Exception:
                            inaccessible_tables.append(
                                f"{table} (exists but no access)"
                            )
                    else:
                        inaccessible_tables.append(f"{table} (does not exist)")
                except Exception as e:
                    inaccessible_tables.append(f"{table} (error: {str(e)})")

            # Display results
            if accessible_tables:
                st.success(f"✅ {len(accessible_tables)} tables are accessible")
                with st.expander("Accessible Tables"):
                    for table in accessible_tables:
                        st.write(f"• {table}")

            if inaccessible_tables:
                st.error(f"❌ {len(inaccessible_tables)} tables have issues")
                with st.expander("Table Issues"):
                    for table in inaccessible_tables:
                        st.write(f"• {table}")

                with st.expander("💡 How to fix table access issues"):
                    st.markdown(
                        """
                    1. **Table doesn't exist**: Check spelling and verify the table exists in your catalog
                    2. **No access**: Contact your admin to grant SELECT permissions on the table
                    3. **Catalog/schema access**: Ensure you have USE CATALOG and USE SCHEMA permissions
                    4. **Service principal permissions**: Verify the app service principal has proper access
                    """
                    )

    def save_table_list(self, tables: List[str]):
        """Save table list as CSV for download"""
        df = pd.DataFrame({"table_name": tables})
        csv = df.to_csv(index=False)

        st.download_button(
            label="⬇️ Download table_names.csv",
            data=csv,
            file_name=f"table_names_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    def render_job_status_section(self):
        """Render job status monitoring section"""
        st.header("📊 Job Status & Monitoring")

        if not st.session_state.workspace_client:
            st.warning("⚠️ Databricks client not initialized. Cannot monitor jobs.")
            return

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("📊 Refresh Job Status"):
                self.refresh_job_status()

        with col2:
            st.session_state.auto_refresh = st.checkbox(
                "🔄 Auto Refresh",
                value=st.session_state.auto_refresh,
                help="Automatically refresh job status every 30 seconds",
            )

        # Display active jobs
        self.display_job_status()

    def refresh_job_status(self):
        """Refresh status of all jobs - orchestrates the status update process"""
        if not self._validate_refresh_prerequisites():
            return

        job_manager = self._initialize_refresh_job_manager()
        if not job_manager:
            return

        self._refresh_all_job_statuses(job_manager)

    def _validate_refresh_prerequisites(self) -> bool:
        """Validate prerequisites for job status refresh"""
        if not st.session_state.job_runs:
            st.info(
                "ℹ️ No active jobs to refresh. Create a job first to monitor status."
            )
            logger.debug(
                f"refresh_job_status called with no active jobs. Session state keys: {list(st.session_state.keys())}"
            )

            # 🔧 ADD DEBUGGING: Show what's in session state
            if hasattr(st.session_state, "job_runs"):
                st.info(
                    f"🔍 Debug: job_runs exists but is empty: {st.session_state.job_runs}"
                )
            else:
                st.info("🔍 Debug: job_runs key doesn't exist in session state")

            return False

        # Check if job manager is available
        if not JOB_MANAGER_AVAILABLE:
            st.error("❌ Job manager not available for status updates")
            st.info("💡 Try restarting the app or check import errors in logs")
            logger.error(f"Job manager unavailable: {JOB_MANAGER_IMPORT_ERROR}")
            return False

        logger.info(
            f"Refresh prerequisites validated. Found {len(st.session_state.job_runs)} job runs"
        )
        st.info(f"🔄 Found {len(st.session_state.job_runs)} job(s) to refresh")
        return True

    def _initialize_refresh_job_manager(self):
        """Initialize job manager for refresh operations"""
        try:
            job_manager = DBXMetaGenJobManager(st.session_state.workspace_client)
            logger.debug("Job manager initialized successfully for refresh")
            return job_manager
        except Exception as e:
            st.error(f"❌ Failed to initialize job manager: {str(e)}")
            logger.error(f"Job manager initialization failed: {str(e)}", exc_info=True)
            return None

    def _refresh_all_job_statuses(self, job_manager):
        """Refresh status for all tracked job runs"""
        try:
            updated_count = 0
            for run_id, job_info in st.session_state.job_runs.items():
                logger.debug(f"Refreshing status for run_id: {run_id}")

                if self._refresh_single_job_status(job_manager, run_id, job_info):
                    updated_count += 1

            st.info(
                f"✅ Refreshed {len(st.session_state.job_runs)} job(s), {updated_count} status changes detected"
            )
            logger.info(
                f"Job refresh completed: {updated_count}/{len(st.session_state.job_runs)} jobs updated"
            )

        except Exception as e:
            st.error(f"❌ Error during job status refresh: {str(e)}")
            logger.error(f"Job status refresh failed: {str(e)}", exc_info=True)

    def _refresh_single_job_status(
        self, job_manager, run_id: int, job_info: Dict[str, Any]
    ) -> bool:
        """Refresh status for a single job run, returns True if status changed"""
        try:
            # Get current status from Databricks
            current_status_info = job_manager.get_run_status(run_id)

            if not current_status_info:
                logger.warning(f"No status info returned for run_id: {run_id}")
                return False

            current_status = current_status_info.get("status")
            if not current_status:
                logger.warning(f"No status field in status info for run_id: {run_id}")
                return False

            old_status = job_info.get("status")

            # Check if status changed
            if current_status != old_status:
                logger.info(
                    f"Job {run_id} status changed: {old_status} → {current_status}"
                )

                # Update job info
                job_info["status"] = current_status
                job_info["last_updated"] = datetime.now()

                # Handle job completion
                if self._is_terminal_status(current_status):
                    job_info["end_time"] = datetime.now()
                logger.info(
                    f"Job {run_id} completed with terminal status: {current_status}"
                )

                return True

            logger.debug(f"Job {run_id} status unchanged: {current_status}")
            return False

        except Exception as e:
            logger.error(
                f"Error refreshing status for run_id {run_id}: {str(e)}", exc_info=True
            )
            return False

    def _is_terminal_status(self, status: str) -> bool:
        """Check if a job status indicates completion (terminal state)"""
        terminal_states = {"SUCCESS", "FAILED", "CANCELED", "TIMEOUT", "TERMINATED"}
        return status in terminal_states

    def stop_job(self, run_id: int):
        """Stop a running job"""
        try:
            st.session_state.workspace_client.jobs.cancel_run(run_id)
            st.session_state.job_runs[run_id]["status"] = "TERMINATED"
            st.success(f"✅ Job run {run_id} stopped successfully")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error stopping job: {str(e)}")

    def display_job_status(self):
        """Display current job status with auto-refresh"""
        # Check if we have any running jobs that need monitoring
        running_jobs = [
            job
            for job in st.session_state.job_runs.values()
            if job["status"] in ["RUNNING", "PENDING"]
        ]

        # Simplified auto-refresh logic (following test_job_manager pattern)
        if st.session_state.auto_refresh and running_jobs:
            st.info(
                f"🔄 Auto-refresh enabled - monitoring {len(running_jobs)} active job(s)"
            )

            # Refresh job status every time the page loads if auto-refresh is on
            current_time = time.time()
            last_refresh = st.session_state.get("last_refresh_time", 0)

            if current_time - last_refresh > st.session_state.refresh_interval:
                self.refresh_job_status()
                st.session_state.last_refresh_time = current_time

                # Check if we still have running jobs after refresh
                running_jobs_after_refresh = [
                    job
                    for job in st.session_state.job_runs.values()
                    if job["status"] in ["RUNNING", "PENDING"]
                ]

                if running_jobs_after_refresh:
                    # Auto-rerun to continue monitoring
                    time.sleep(2)  # Small delay for better UX
                    st.rerun()
                else:
                    st.success("🎉 All jobs completed!")
                    st.session_state.auto_refresh = False

        elif st.session_state.auto_refresh and not running_jobs:
            st.info("🔄 Auto-refresh enabled - no active jobs to monitor")
            st.session_state.auto_refresh = False  # Auto-disable when no jobs

        if not st.session_state.job_runs:
            st.info("ℹ️ No jobs running")
            return

        # Job status header with manual refresh button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("📊 Job Status")
        with col2:
            if st.button("🔄 Refresh Now", help="Manually refresh job status"):
                with st.spinner("Refreshing job status..."):
                    self.refresh_job_status()
                st.success("✅ Status refreshed!")
                st.rerun()

        for run_id, job_info in st.session_state.job_runs.items():
            # Calculate progress for running jobs
            progress = 0
            if job_info["status"] == "SUCCESS":
                progress = 100
            elif job_info["status"] == "RUNNING":
                # Estimate progress based on time elapsed (rough estimate)
                elapsed = (datetime.now() - job_info["start_time"]).total_seconds()
                progress = min(80, (elapsed / 600) * 100)  # Max 80% for running jobs
            elif job_info["status"] == "FAILED":
                progress = 0

            with st.expander(
                f"🔧 {job_info['job_name']} (Run ID: {run_id})",
                expanded=job_info["status"] == "RUNNING",
            ):
                # Progress bar for running jobs
                if job_info["status"] in ["RUNNING", "SUCCESS"]:
                    st.progress(progress / 100, text=f"Progress: {progress:.0f}%")

                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    status_color = {
                        "RUNNING": "🟡",
                        "SUCCESS": "🟢",
                        "FAILED": "🔴",
                        "TERMINATED": "🟠",
                        "PENDING": "🔵",
                    }.get(job_info["status"], "⚪")

                    st.write(f"**Status:** {status_color} {job_info['status']}")
                    st.write(f"**Mode:** {job_info['mode']}")
                    st.write(f"**Tables:** {len(job_info['tables'])}")

                with col2:
                    st.write(
                        f"**Started:** {job_info['start_time'].strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    if "end_time" in job_info:
                        st.write(
                            f"**Ended:** {job_info['end_time'].strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        # Calculate duration
                        duration = job_info["end_time"] - job_info["start_time"]
                        st.write(f"**Duration:** {str(duration).split('.')[0]}")
                    elif job_info["status"] == "RUNNING":
                        elapsed = datetime.now() - job_info["start_time"]
                        st.write(f"**Elapsed:** {str(elapsed).split('.')[0]}")

                with col3:
                    if st.button(f"🔗 View in Databricks", key=f"view_{run_id}"):
                        databricks_url = f"{st.session_state.config['host']}#job/{job_info['job_id']}/run/{run_id}"
                        st.markdown(f"[Open Job Run]({databricks_url})")

                    # Add a stop job button for running jobs
                    if job_info["status"] == "RUNNING":
                        if st.button(f"⏹️ Stop Job", key=f"stop_{run_id}"):
                            self.stop_job(run_id)

    def render_results_viewer(self):
        """Render results viewing section"""
        st.header("📊 Results Viewer")

        tab1, tab2, tab3 = st.tabs(
            ["📄 Generated Files", "🏷️ Table Metadata", "📈 Statistics"]
        )

        with tab1:
            self.render_file_viewer()

        with tab2:
            self.render_metadata_viewer()

        with tab3:
            self.render_statistics()

    def render_file_viewer(self):
        """Render file viewing section"""
        st.subheader("📁 Generated Files")

        # File upload for viewing results
        uploaded_result = st.file_uploader(
            "Upload Generated File",
            type=["tsv", "sql", "csv"],
            help="Upload TSV, SQL, or CSV files generated by dbxmetagen",
        )

        if uploaded_result:
            file_type = uploaded_result.name.split(".")[-1].lower()

            try:
                if file_type in ["tsv", "csv"]:
                    separator = "\t" if file_type == "tsv" else ","
                    df = pd.read_csv(uploaded_result, sep=separator)

                    st.success(f"✅ Loaded {len(df)} rows from {uploaded_result.name}")

                    # Display with filters
                    col1, col2 = st.columns([1, 3])

                    with col1:
                        if "table_name" in df.columns:
                            selected_tables = st.multiselect(
                                "Filter by Table",
                                options=df["table_name"].unique(),
                                default=df["table_name"].unique()[:5],
                            )
                            df = (
                                df[df["table_name"].isin(selected_tables)]
                                if selected_tables
                                else df
                            )

                    with col2:
                        st.dataframe(df, use_container_width=True, height=400)

                    # Download filtered data
                    if not df.empty:
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            "⬇️ Download Filtered Data",
                            data=csv_data,
                            file_name=f"filtered_{uploaded_result.name}",
                            mime="text/csv",
                        )

                elif file_type == "sql":
                    content = uploaded_result.read().decode("utf-8")
                    st.success(f"✅ Loaded SQL file: {uploaded_result.name}")

                    # Display SQL with syntax highlighting
                    st.code(content, language="sql")

                    # Option to download
                    st.download_button(
                        "⬇️ Download SQL",
                        data=content,
                        file_name=uploaded_result.name,
                        mime="text/sql",
                    )

            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

    def render_metadata_viewer(self):
        """Render metadata viewing section"""
        st.subheader("🏷️ Table Metadata Viewer")

        if not st.session_state.workspace_client:
            st.warning("⚠️ Databricks client not initialized")
            return

        # Table input for metadata viewing
        table_name = st.text_input(
            "Table Name",
            placeholder="catalog.schema.table",
            help="Enter fully qualified table name to view metadata",
        )

        if table_name and st.button("🔍 Get Metadata"):
            self.display_table_metadata(table_name)

    def display_table_metadata(self, table_name: str):
        """Display metadata for a specific table"""
        try:
            with st.spinner(f"🔍 Fetching metadata for {table_name}..."):
                # This would need to be implemented based on your specific metadata storage
                # For now, showing a placeholder structure

                st.success(f"✅ Metadata for {table_name}")

                # Placeholder metadata structure
                metadata = {
                    "table_comment": "AI-generated table description would appear here",
                    "columns": [
                        {
                            "name": "column1",
                            "type": "string",
                            "comment": "AI-generated column description",
                            "tags": ["PII"],
                        },
                        {
                            "name": "column2",
                            "type": "int",
                            "comment": "Another AI-generated description",
                            "tags": ["Non-PII"],
                        },
                    ],
                }

                # Display table comment
                if metadata.get("table_comment"):
                    st.info(f"**Table Description:** {metadata['table_comment']}")

                # Display column metadata
                if metadata.get("columns"):
                    st.subheader("📋 Column Metadata")

                    cols_df = pd.DataFrame(metadata["columns"])
                    st.dataframe(cols_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error fetching metadata: {str(e)}")

    def render_statistics(self):
        """Render statistics and metrics"""
        st.subheader("📈 Processing Statistics")

        # Placeholder for statistics
        if st.session_state.job_runs:
            total_jobs = len(st.session_state.job_runs)
            completed_jobs = sum(
                1
                for job in st.session_state.job_runs.values()
                if job["status"] == "SUCCESS"
            )
            failed_jobs = sum(
                1
                for job in st.session_state.job_runs.values()
                if job["status"] == "FAILED"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Jobs", total_jobs)
            with col2:
                st.metric("Completed", completed_jobs)
            with col3:
                st.metric("Failed", failed_jobs)
            with col4:
                st.metric(
                    "Success Rate",
                    (
                        f"{(completed_jobs/total_jobs*100):.1f}%"
                        if total_jobs > 0
                        else "0%"
                    ),
                )
        else:
            st.info("ℹ️ No job statistics available yet")

    def render_metadata_review(self):
        """Render metadata review and editing interface"""
        st.header("✏️ Metadata Review & Editing")
        st.markdown("Review and modify AI-generated metadata before applying to tables")

        # File upload section
        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_metadata = st.file_uploader(
                "📁 Upload Generated Metadata File",
                type=["tsv", "csv"],
                help="Upload TSV/CSV file containing generated metadata for review",
            )

        with col2:
            if st.button("📂 Load from Volume"):
                self.load_metadata_from_volume()

        if uploaded_metadata:
            self.review_uploaded_metadata(uploaded_metadata)

    def load_metadata_from_volume(self) -> None:
        """Load metadata files from volume for review."""
        if not self._validate_workspace_client():
            return

        try:
            with st.spinner("📂 Loading metadata files from volume..."):
                volume_path = self._build_volume_path()
                self._display_volume_header(volume_path)

                metadata_files = self._list_and_filter_volume_files(volume_path)

                if metadata_files:
                    self._display_file_selection_interface(metadata_files)
                else:
                    self._show_no_files_found_message()

        except Exception as e:
            st.error(f"❌ Error loading files from volume: {str(e)}")
            logger.error(f"load_metadata_from_volume failed: {str(e)}", exc_info=True)
            handle_databricks_error(e, "volume file listing")

    def _build_volume_path(self) -> str:
        """
        Build the volume path from configuration.

        Returns:
            Complete volume path string
        """
        config = self._extract_volume_config()
        return f"/Volumes/{config['catalog_name']}/{config['schema_name']}/{config['volume_name']}/"

    def _display_volume_header(self, volume_path: str) -> None:
        """
        Display volume selection header information.

        Args:
            volume_path: Path to the volume
        """
        st.subheader("📁 Select Metadata File from Volume")
        st.info(f"📍 Volume path: {volume_path}")

    def _list_and_filter_volume_files(self, volume_path: str) -> List[Dict[str, Any]]:
        """
        List files in volume and filter for metadata files.

        Args:
            volume_path: Path to the volume

        Returns:
            List of metadata file information dictionaries
        """
        try:
            files = list(
                st.session_state.workspace_client.files.list_directory_contents(
                    volume_path
                )
            )
            return self._filter_metadata_files(files)

        except Exception as file_error:
            self._handle_volume_access_error(file_error, volume_path)
            return []

    def _filter_metadata_files(self, files) -> List[Dict[str, Any]]:
        """
        Filter files for metadata file types (TSV, CSV).

        Args:
            files: List of file objects from Databricks API

        Returns:
            List of filtered metadata files with formatted information
        """
        metadata_files = []

        for file_info in files:
            if file_info.name.lower().endswith((".tsv", ".csv")):
                metadata_files.append(
                    {
                        "name": file_info.name,
                        "path": file_info.path,
                        "size": file_info.size_bytes,
                        "modified": file_info.last_modified,
                    }
                )

        # Sort by last modified (newest first)
        return sorted(metadata_files, key=lambda x: x["modified"] or 0, reverse=True)

    def _display_file_selection_interface(
        self, metadata_files: List[Dict[str, Any]]
    ) -> None:
        """
        Display file selection interface for metadata files.

        Args:
            metadata_files: List of metadata file information
        """
        st.success(f"✅ Found {len(metadata_files)} metadata file(s)")

        file_options, file_details = self._prepare_file_selection_data(metadata_files)

        selected_file_display = st.selectbox(
            "Choose a metadata file:",
            options=file_options,
            help="Select a TSV or CSV file containing generated metadata",
        )

        if selected_file_display and st.button("📥 Load Selected File", type="primary"):
            selected_file = file_details[selected_file_display]
            self._load_file_from_volume(selected_file)

    def _prepare_file_selection_data(
        self, metadata_files: List[Dict[str, Any]]
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """
        Prepare data structures for file selection interface.

        Args:
            metadata_files: List of metadata file information

        Returns:
            Tuple of (file_options_list, file_details_dict)
        """
        file_options = []
        file_details = {}

        for file_info in metadata_files:
            size_str = self._format_file_size(file_info["size"])
            modified_str = self._format_file_timestamp(file_info["modified"])

            display_name = f"{file_info['name']} ({size_str}, {modified_str})"
            file_options.append(display_name)
            file_details[display_name] = file_info

        return file_options, file_details

    def _format_file_timestamp(self, timestamp: Optional[int]) -> str:
        """
        Format file timestamp for display.

        Args:
            timestamp: Unix timestamp in milliseconds

        Returns:
            Formatted timestamp string
        """
        if timestamp:
            return datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d %H:%M")
        return "Unknown"

    def _show_no_files_found_message(self) -> None:
        """Display message when no metadata files are found in volume."""
        st.warning("⚠️ No metadata files (TSV/CSV) found in the volume")

        with st.expander("💡 Expected file types"):
            st.markdown(
                """
            Looking for files with these extensions:
            - `.tsv` - Tab-separated values (preferred)
            - `.csv` - Comma-separated values
            
            Make sure your metadata generation jobs have completed and saved files to this volume.
            """
            )

        st.info(
            "📤 Alternatively, you can upload a file using the file uploader above."
        )

    def _handle_volume_access_error(
        self, file_error: Exception, volume_path: str
    ) -> None:
        """
        Handle errors when accessing volume files.

        Args:
            file_error: Exception that occurred
            volume_path: Path that was being accessed
        """
        error_msg = str(file_error).lower()

        if "not found" in error_msg or "does not exist" in error_msg:
            st.error(f"❌ Volume not found: {volume_path}")
            self._show_volume_troubleshooting_guide(volume_path)
        else:
            st.error(f"❌ Error accessing volume: {str(file_error)}")
            handle_databricks_error(file_error, "volume access")

    def _show_volume_troubleshooting_guide(self, volume_path: str) -> None:
        """
        Display troubleshooting guide for volume access issues.

        Args:
            volume_path: Volume path that failed
        """
        config = self._extract_volume_config()

        with st.expander("💡 How to fix volume issues"):
            st.markdown(
                f"""
            **Volume path not found:** `{volume_path}`
            
            **Possible solutions:**
            1. **Check volume exists**: Verify the volume exists in your catalog
            2. **Check permissions**: Ensure the service principal has access to the volume
            3. **Verify path**: Confirm catalog, schema, and volume names are correct
            4. **Create volume**: Create the volume if it doesn't exist:
                ```sql
        CREATE VOLUME {config['catalog_name']}.{config['schema_name']}.{config['volume_name']};
                        ```
                    """
            )

    def _format_file_size(self, size_bytes):
        """Format file size in human-readable format"""
        if size_bytes is None:
            return "Unknown size"

        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def _load_file_from_volume(self, file_info: Dict[str, Any]) -> None:
        """
        Load a specific file from volume and process it for review.

        Args:
            file_info: Dictionary containing file metadata (name, path, size, modified)
        """
        try:
            with st.spinner(f"📥 Loading {file_info['name']}..."):
                df = self._download_and_parse_volume_file(file_info)
                self._show_volume_load_success(df, file_info["name"])
                self._store_metadata_in_session(df, file_info)
                self._display_volume_file_review(df, file_info["name"])

        except Exception as e:
            st.error(f"❌ Error loading file {file_info['name']}: {str(e)}")
            logger.error(
                f"_load_file_from_volume failed for {file_info['name']}: {str(e)}",
                exc_info=True,
            )
            handle_databricks_error(e, "file loading")

    def _download_and_parse_volume_file(
        self, file_info: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Download file from volume and parse as DataFrame.

        Args:
            file_info: Dictionary containing file metadata

        Returns:
            DataFrame containing the parsed file data
        """
        # Download file content
        file_content = st.session_state.workspace_client.files.download(
            file_info["path"]
        )
        content_str = file_content.contents.decode("utf-8")

        # Parse content
        return self._parse_file_content(content_str, file_info["name"])

    def _parse_file_content(self, content_str: str, filename: str) -> pd.DataFrame:
        """
        Parse file content string as DataFrame with appropriate separator.

        Args:
            content_str: File content as string
            filename: Name of the file for separator detection

        Returns:
            DataFrame containing parsed data
        """
        from io import StringIO

        file_obj = StringIO(content_str)
        separator = self._detect_file_separator(filename)

        return pd.read_csv(file_obj, sep=separator)

    def _detect_file_separator(self, filename: str) -> str:
        """
        Detect file separator based on file extension.

        Args:
            filename: Name of the file

        Returns:
            Separator character (tab or comma)
        """
        return "\t" if filename.lower().endswith(".tsv") else ","

    def _show_volume_load_success(self, df: pd.DataFrame, filename: str) -> None:
        """
        Display success message for volume file loading.

        Args:
            df: Loaded DataFrame
            filename: Name of loaded file
        """
        st.success(f"✅ Successfully loaded {len(df)} records from {filename}")

    def _store_metadata_in_session(
        self, df: pd.DataFrame, file_info: Dict[str, Any]
    ) -> None:
        """
        Store loaded metadata in session state for later use.

        Args:
            df: Loaded DataFrame
            file_info: Dictionary containing file metadata
        """
        st.session_state.current_metadata = {
            "data": df,
            "filename": file_info["name"],
            "source": "volume",
            "path": file_info["path"],
        }

    def _display_volume_file_review(self, df: pd.DataFrame, filename: str) -> None:
        """
        Display the review interface for volume-loaded file.

        Args:
            df: DataFrame to review
            filename: Name of the file being reviewed
        """
        st.markdown("---")
        st.subheader(f"📝 Reviewing: {filename}")
        self._review_loaded_metadata(df, filename)

    def _review_loaded_metadata(self, df: pd.DataFrame, filename: str) -> None:
        """
        Review loaded metadata with filtering and editing capabilities.

        Args:
            df: DataFrame containing metadata to review
            filename: Name of the source file
        """
        try:
            self._show_metadata_summary(df)
            filtered_df = self._apply_metadata_filters(df)
            edited_df = self._create_metadata_editor(filtered_df, key_prefix="volume")
            self._show_metadata_action_buttons(edited_df, filename, key_prefix="volume")

        except Exception as e:
            st.error(f"❌ Error processing metadata: {str(e)}")
            logger.error(f"_review_loaded_metadata failed: {str(e)}", exc_info=True)

    def _show_metadata_summary(self, df: pd.DataFrame) -> None:
        """
        Display summary statistics for the loaded metadata.

        Args:
            df: DataFrame containing metadata
        """
        with st.expander("📊 Metadata Summary"):
            col1, col2, col3 = st.columns(3)

            with col1:
                if "table_name" in df.columns:
                    st.metric("Tables", df["table_name"].nunique())

            with col2:
                if "column_name" in df.columns:
                    st.metric("Columns", len(df))

            with col3:
                if "mode" in df.columns:
                    mode_value = df["mode"].iloc[0] if not df.empty else "Unknown"
                    st.metric("Processing Mode", mode_value)

    def _apply_metadata_filters(
        self, df: pd.DataFrame, key_prefix: str = "volume"
    ) -> pd.DataFrame:
        """
        Apply filters to the metadata DataFrame.

        Args:
            df: DataFrame to filter
            key_prefix: Unique key prefix for Streamlit widgets

        Returns:
            Filtered DataFrame
        """
        st.subheader("🔍 Filter & Select")
        col1, col2 = st.columns(2)

        filtered_df = df.copy()

        with col1:
            filtered_df = self._apply_table_filter(
                filtered_df, f"{key_prefix}_table_filter"
            )

        with col2:
            filtered_df = self._apply_column_search(
                filtered_df, f"{key_prefix}_column_search"
            )

        return filtered_df

    def _apply_table_filter(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        """
        Apply table name filter to DataFrame.

        Args:
            df: DataFrame to filter
            key: Unique key for Streamlit widget

        Returns:
            Filtered DataFrame
        """
        if "table_name" not in df.columns:
            return df

        unique_tables = sorted(df["table_name"].unique())
        default_selection = (
            unique_tables[:5] if len(unique_tables) > 5 else unique_tables
        )

        selected_tables = st.multiselect(
            "Filter by Tables",
            options=unique_tables,
            default=default_selection,
            key=key,
        )

        if selected_tables:
            return df[df["table_name"].isin(selected_tables)]
        return df

    def _apply_column_search(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        """
        Apply column name search filter to DataFrame.

        Args:
            df: DataFrame to filter
            key: Unique key for Streamlit widget

        Returns:
            Filtered DataFrame
        """
        if "column_name" not in df.columns:
            return df

        search_term = st.text_input(
            "Search Columns",
            placeholder="Enter column name or pattern...",
            help="Filter columns by name",
            key=key,
        )

        if search_term:
            return df[df["column_name"].str.contains(search_term, case=False, na=False)]
        return df

    def _create_metadata_editor(
        self, df: pd.DataFrame, key_prefix: str = "metadata"
    ) -> pd.DataFrame:
        """
        Create an editable interface for metadata.

        Args:
            df: DataFrame to edit
            key_prefix: Unique key prefix for the editor

        Returns:
            Edited DataFrame
        """
        st.subheader("✏️ Edit Metadata")
        st.info("💡 Double-click on cells to edit. Focus on comment and tag columns.")

        if df.empty:
            st.warning("⚠️ No data to edit")
            return df

        column_config = self._get_metadata_column_config()
        disabled_columns = self._get_disabled_columns(df)

        return st.data_editor(
            df,
            use_container_width=True,
            height=400,
            column_config=column_config,
            disabled=disabled_columns,
            key=f"{key_prefix}_editor",
        )

    def _get_metadata_column_config(self) -> Dict[str, st.column_config.TextColumn]:
        """
        Get column configuration for the metadata editor.

        Returns:
            Dictionary of column configurations
        """
        return {
            "generated_comment": st.column_config.TextColumn(
                "Generated Comment",
                help="AI-generated description",
                max_chars=500,
            ),
            "tags": st.column_config.TextColumn(
                "Tags",
                help="Comma-separated tags (e.g., PII, Non-PII)",
            ),
            "table_comment": st.column_config.TextColumn(
                "Table Comment",
                help="Table-level description",
                max_chars=500,
            ),
        }

    def _get_disabled_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of columns that should be disabled in the editor.

        Args:
            df: DataFrame to check for columns

        Returns:
            List of column names to disable
        """
        base_disabled = ["table_name", "column_name"]
        if "data_type" in df.columns:
            base_disabled.append("data_type")
        return base_disabled

    def _show_metadata_action_buttons(
        self, edited_df: pd.DataFrame, filename: str, key_prefix: str = "metadata"
    ) -> None:
        """
        Display action buttons for metadata operations.

        Args:
            edited_df: Edited DataFrame
            filename: Source filename
            key_prefix: Unique key prefix for buttons
        """
        st.subheader("🚀 Actions")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(
                "💾 Save Edited Metadata", type="primary", key=f"{key_prefix}_save"
            ):
                self.save_edited_metadata(edited_df, filename)

        with col2:
            if st.button("🔄 Generate DDL", key=f"{key_prefix}_generate_ddl"):
                self.generate_ddl_from_metadata(edited_df)

        with col3:
            if st.button(
                "✅ Apply to Tables", type="secondary", key=f"{key_prefix}_apply"
            ):
                self.apply_metadata_to_tables(edited_df)

    def review_uploaded_metadata(self, uploaded_file) -> None:
        """
        Process and allow editing of uploaded metadata.

        Args:
            uploaded_file: Streamlit uploaded file object
        """
        try:
            df = self._load_uploaded_file(uploaded_file)
            self._show_upload_success(df, uploaded_file.name)

            # Use the same review interface as volume-loaded metadata
            self._show_metadata_summary(df)
            filtered_df = self._apply_metadata_filters(df, key_prefix="upload")
            edited_df = self._create_metadata_editor(filtered_df, key_prefix="upload")
            self._show_metadata_action_buttons(
                edited_df, uploaded_file.name, key_prefix="upload"
            )

        except Exception as e:
            st.error(f"❌ Error processing metadata file: {str(e)}")
            logger.error(f"review_uploaded_metadata failed: {str(e)}", exc_info=True)

    def _load_uploaded_file(self, uploaded_file) -> pd.DataFrame:
        """
        Load uploaded file into DataFrame with appropriate separator detection.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            DataFrame containing the loaded data
        """
        file_ext = uploaded_file.name.split(".")[-1].lower()
        separator = "\t" if file_ext == "tsv" else ","
        return pd.read_csv(uploaded_file, sep=separator)

    def _show_upload_success(self, df: pd.DataFrame, filename: str) -> None:
        """
        Display success message for uploaded file.

        Args:
            df: Loaded DataFrame
            filename: Name of uploaded file
        """
        st.success(f"✅ Loaded {len(df)} metadata records from {filename}")

    def save_edited_metadata(self, df: pd.DataFrame, original_filename: str):
        """Save edited metadata for download"""
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"reviewed_{timestamp}_{original_filename}"

            # Convert to CSV/TSV based on original format
            if original_filename.endswith(".tsv"):
                csv_data = df.to_csv(sep="\t", index=False)
                mime_type = "text/tab-separated-values"
            else:
                csv_data = df.to_csv(index=False)
                mime_type = "text/csv"

            st.download_button(
                label="⬇️ Download Reviewed Metadata",
                data=csv_data,
                file_name=new_filename,
                mime=mime_type,
            )

            st.success("✅ Reviewed metadata ready for download!")

        except Exception as e:
            st.error(f"❌ Error saving metadata: {str(e)}")

    def generate_ddl_from_metadata(self, df: pd.DataFrame):
        """Generate DDL statements from reviewed metadata"""
        try:
            ddl_statements = []

            # Group by table for table comments
            if "table_comment" in df.columns:
                for table_name, group in df.groupby("table_name"):
                    table_comment = group["table_comment"].iloc[0]
                    if pd.notna(table_comment) and table_comment.strip():
                        # Escape single quotes in the comment
                        escaped_comment = table_comment.replace("'", "''")
                        ddl_statements.append(
                            f"COMMENT ON TABLE `{table_name}` IS '{escaped_comment}';"
                        )

            # Add column comments
            if "generated_comment" in df.columns:
                for _, row in df.iterrows():
                    if (
                        pd.notna(row.get("generated_comment"))
                        and row.get("generated_comment", "").strip()
                    ):
                        comment = row["generated_comment"].replace("'", "''")
                        ddl_statements.append(
                            f"ALTER TABLE `{row['table_name']}` ALTER COLUMN `{row['column_name']}` COMMENT '{comment}';"
                        )

            # Display generated DDL
            if ddl_statements:
                ddl_text = "\n".join(ddl_statements)

                st.subheader("🔧 Generated DDL")
                st.code(ddl_text, language="sql")

                # Download DDL
                st.download_button(
                    label="⬇️ Download DDL",
                    data=ddl_text,
                    file_name=f"metadata_ddl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql",
                    mime="text/sql",
                )
            else:
                st.warning("⚠️ No comments found to generate DDL")

        except Exception as e:
            st.error(f"❌ Error generating DDL: {str(e)}")

    def apply_metadata_to_tables(self, df: pd.DataFrame) -> None:
        """
        Apply metadata directly to tables using sync_reviewed_ddl notebook.

        Args:
            df: DataFrame containing the metadata to apply
        """
        if not self._validate_workspace_client():
            return

        try:
            with st.spinner("🚀 Applying metadata to tables..."):
                config = self._extract_volume_config()
                reviewed_filename = self._generate_reviewed_filename()

                self._show_metadata_preview(df)
                self._show_manual_instructions(config, reviewed_filename)
                self._show_sync_job_option(df, reviewed_filename)

        except Exception as e:
            st.error(f"❌ Error applying metadata: {str(e)}")
            logger.error(f"apply_metadata_to_tables failed: {str(e)}", exc_info=True)

    def _validate_workspace_client(self) -> bool:
        """
        Validate that the workspace client is initialized.

        Returns:
            True if client is initialized, False otherwise
        """
        if not st.session_state.workspace_client:
            st.warning("⚠️ Databricks client not initialized")
            st.info("💡 Check your authentication settings in the Configuration tab")
            return False
        return True

    def _extract_volume_config(self) -> Dict[str, str]:
        """
        Extract volume configuration from session state.

        Returns:
            Dictionary containing catalog, schema, and volume names
        """
        return {
            "catalog_name": st.session_state.config.get("catalog_name", "dbxmetagen"),
            "schema_name": st.session_state.config.get(
                "schema_name", "metadata_results"
            ),
            "volume_name": st.session_state.config.get(
                "volume_name", "generated_metadata"
            ),
        }

    def _generate_reviewed_filename(self) -> str:
        """
        Generate a timestamped filename for reviewed metadata.

        Returns:
            Filename with timestamp
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"reviewed_metadata_{timestamp}.tsv"

    def _show_metadata_preview(self, df: pd.DataFrame) -> None:
        """
        Display a preview of metadata that will be applied.

        Args:
            df: DataFrame containing metadata to preview
        """
        st.subheader("📋 Preview: Changes to Apply")
        st.info(
            f"Will apply metadata from {len(df)} records to {df['table_name'].nunique()} tables"
        )

    def _show_manual_instructions(self, config: Dict[str, str], filename: str) -> None:
        """
        Display manual instructions for applying metadata.

        Args:
            config: Volume configuration dictionary
            filename: Name of the reviewed metadata file
        """
        st.warning("🚧 **Manual Step Required:**")
        st.markdown(
            f"""
                To apply the reviewed metadata:
                1. **Download the reviewed metadata** using the "Save Edited Metadata" button above
        2. **Upload it to your volume** at: `/Volumes/{config['catalog_name']}/{config['schema_name']}/{config['volume_name']}/`
                3. **Run the sync_reviewed_ddl notebook** with the uploaded file path
                
                Or create a job to automate this process using the sync_reviewed_ddl notebook.
                """
        )

    def _show_sync_job_option(self, df: pd.DataFrame, filename: str) -> None:
        """
        Display option to create a sync job for metadata application.

        Args:
            df: DataFrame containing metadata
            filename: Name of the reviewed metadata file
        """
        if st.button("🚀 Create Sync Job"):
            self.create_sync_metadata_job(df, filename)

    def create_sync_metadata_job(self, df: pd.DataFrame, filename: str):
        """Create a job to sync reviewed metadata using sync_reviewed_ddl notebook"""
        try:
            # Job parameters
            job_parameters = {
                "catalog_name": st.session_state.config.get(
                    "catalog_name", "dbxmetagen"
                ),
                "schema_name": st.session_state.config.get(
                    "schema_name", "metadata_results"
                ),
                "volume_name": st.session_state.config.get(
                    "volume_name", "generated_metadata"
                ),
                "reviewed_metadata_file": filename,
                "apply_changes": "true",
            }

            job_name = (
                f"sync_reviewed_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            # Create job configuration for sync_reviewed_ddl notebook
            job_config = JobSettings(
                environments=[
                    JobEnvironment(
                        environment_key="default_python",
                        spec=Environment(
                            environment_version="1",
                            # dependencies=["./requirements.txt"],
                        ),
                    )
                ],
                performance_target=PerformanceTarget.PERFORMANCE_OPTIMIZED,
                name=job_name,
                tasks=[
                    Task(
                        task_key="sync_metadata",
                        notebook_task=NotebookTask(
                            notebook_path="./notebooks/sync_reviewed_ddl",
                            base_parameters=job_parameters,
                        ),
                        # "job_cluster_key": "sync_cluster",
                    )
                ],
                # job_clusters=[
                #     JobCluster(
                #         job_cluster_key="sync_cluster",
                #         new_cluster=ClusterSpec(
                #             spark_version="16.4.x-cpu-ml-scala2.12",
                #             node_type_id="Standard_D3_v2",
                #             num_workers=1,  # Small cluster for metadata sync
                #             runtime_engine="STANDARD",
                #         ),
                #     )
                # ],
            )

            # Create the job
            job = st.session_state.workspace_client.jobs.create(**job_config.as_dict())

            st.success(f"✅ Sync job created! Job ID: {job.job_id}")
            st.info(
                "📋 Upload your reviewed metadata file to the volume and then run this job."
            )

        except Exception as e:
            st.error(f"❌ Error creating sync job: {str(e)}")

    def run(self):
        """Main app execution"""
        # Header
        st.title("🏷️ DBX MetaGen")
        st.markdown("### AI-Powered Metadata Generation for Databricks Tables")

        # Sidebar configuration
        self.render_sidebar_config()

        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📋 Tables & Jobs", "📊 Results", "✏️ Review Metadata", "❓ Help"]
        )

        with tab1:
            self.render_unified_table_management()
            st.markdown("---")
            self.render_job_status_section()

        with tab2:
            self.render_results_viewer()

        with tab3:
            self.render_metadata_review()

        with tab4:
            self.render_help()

    def render_help(self):
        """Render help and documentation"""
        st.header("❓ Help & Documentation")

        with st.expander("🚀 Getting Started", expanded=True):
            st.markdown(
                """
            1. **Configure Settings**: Use the sidebar to set your Databricks catalog, host, and processing options
            2. **Select Tables**: Either manually enter table names or upload a CSV file with table names
            3. **Create Job**: Configure and create a metadata generation job
            4. **Monitor Progress**: Watch job status and get notified when complete
            5. **View Results**: Browse generated metadata, download files, and review table comments
            """
            )

        with st.expander("⚙️ Configuration Options"):
            st.markdown(
                """
            - **Catalog Name**: Target catalog for storing metadata results
            - **Allow Data**: Whether to include actual data samples in LLM processing
            - **Sample Size**: Number of data rows to sample per column (0 = no data sampling)
            - **Mode**: Choose between generating comments, identifying PII, or both
            - **Apply DDL**: ⚠️ **WARNING** - This will directly modify your tables
            """
            )

        with st.expander("📁 File Formats"):
            st.markdown(
                """
            - **table_names.csv**: CSV file with 'table_name' column containing fully qualified table names
            - **Generated TSV**: Tab-separated file with metadata results
            - **Generated SQL**: DDL statements to apply metadata to tables
            """
            )

        with st.expander("🔧 Troubleshooting"):
            st.markdown(
                """
            - **Authentication Issues**: Ensure Databricks token and host are properly configured
            - **Table Access**: Verify you have read access to the tables you want to process
            - **Job Failures**: Check job logs in Databricks workspace for detailed error information
            - **Large Tables**: Consider reducing sample size for very large tables
            """
            )


# Run the app
if __name__ == "__main__":
    # Add emergency debugging section
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Debug Information")

    if st.sidebar.button("Show Debug Info"):
        st.sidebar.write("**System Status:**")
        st.sidebar.write(f"- Job Manager Available: {JOB_MANAGER_AVAILABLE}")
        if not JOB_MANAGER_AVAILABLE:
            st.sidebar.write(f"- Import Error: {JOB_MANAGER_IMPORT_ERROR}")

        if "workspace_client" in st.session_state:
            st.sidebar.write(
                f"- Workspace Client: ✅ {type(st.session_state.workspace_client)}"
            )
        else:
            st.sidebar.write("- Workspace Client: ❌ Not initialized")

        if "config" in st.session_state:
            st.sidebar.write(f"- Config: ✅ {len(st.session_state.config)} keys")
            st.sidebar.write(f"- Config keys: {list(st.session_state.config.keys())}")
        else:
            st.sidebar.write("- Config: ❌ Not loaded")

        if "debug_logs" in st.session_state and st.session_state.debug_logs:
            st.sidebar.write(
                f"- Debug Logs: {len(st.session_state.debug_logs)} entries"
            )
            with st.sidebar.expander("Recent Logs"):
                for log in st.session_state.debug_logs[-5:]:
                    st.sidebar.text(log)
        else:
            st.sidebar.write("- Debug Logs: Empty")

        # Add log viewing instructions
        st.sidebar.markdown("---")
        st.sidebar.write("**📋 View Full Logs:**")
        st.sidebar.write("Add `/logz` to your app URL")
        st.sidebar.write("Example: `your-app-url.com/logz`")

        # Log this debug info request
        logger.info("🔍 Debug info requested by user")
        logger.info(f"Job Manager Available: {JOB_MANAGER_AVAILABLE}")
        logger.info(
            f"Workspace Client: {type(st.session_state.workspace_client) if 'workspace_client' in st.session_state else 'Not initialized'}"
        )
        logger.info(
            f"Config Keys: {list(st.session_state.config.keys()) if 'config' in st.session_state else 'Not loaded'}"
        )

    app = DBXMetaGenApp()
    app.run()
