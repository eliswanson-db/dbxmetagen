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
from typing import Dict, Any, Optional, List
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
                except Exception:
                    pass

                # Method 2: Try query parameters (sometimes user token is passed this way)
                if not user_token:
                    try:
                        query_params = st.query_params
                        user_token = query_params.get("token") or query_params.get(
                            "access_token"
                        )
                    except Exception:
                        pass

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
        """Load default configuration from variables.yml"""
        # Check if config is already loaded in session state
        if "default_config_loaded" in st.session_state:
            return st.session_state.default_config_loaded

        config_paths = ["variables.yml", "../variables.yml", "../../variables.yml"]

        for config_path in config_paths:
            try:
                with open(config_path, "r") as f:
                    variables = yaml.safe_load(f)

                if "variables" in variables:
                    # Extract default values from the variables.yml structure
                    default_config = {}
                    for key, config in variables["variables"].items():
                        default_config[key] = config.get("default")

                    # Get host from environment if available, otherwise use from config
                    if "host" in default_config and st.session_state.get(
                        "databricks_host"
                    ):
                        default_config["host"] = st.session_state.databricks_host

                    # Sanitize unresolved placeholders for owner_user
                    owner_user_val = (default_config.get("owner_user") or "").strip()
                    if owner_user_val.startswith("${") and owner_user_val.endswith("}"):
                        default_config["owner_user"] = ""

                    # Cache the loaded config in session state
                    st.session_state.default_config_loaded = default_config
                    st.info(f"✅ Configuration loaded from {config_path}")
                    return default_config
                else:
                    st.warning(f"⚠️ {config_path} found but has unexpected format")

            except FileNotFoundError:
                continue  # Try next path
            except Exception as e:
                st.warning(f"⚠️ Error loading {config_path}: {str(e)}")
                continue

        # If no config file found, show info and use defaults
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
        if not st.session_state.config:
            st.session_state.config = self.load_default_config()

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
        """Render unified table management and job creation section"""
        st.header("📋 Table Management & Job Creation")

        # Unified table input section
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
                    # Update the manual entry with CSV data
                    st.session_state.selected_tables = tables_from_csv
                    st.rerun()

        # Parse tables from input
        if table_names_input.strip():
            tables = [t.strip() for t in table_names_input.split("\n") if t.strip()]
            st.session_state.selected_tables = tables
        else:
            tables = st.session_state.get("selected_tables", [])

        # Action buttons row
        if tables:
            st.info(f"📊 {len(tables)} table(s) selected")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button("🔍 Validate Tables", type="secondary"):
                    self.validate_tables(tables)

            with col2:
                if st.button("💾 Download CSV"):
                    self.save_table_list(tables)

            with col3:
                # TODO: Fix this. For now, we're subbing in Run Job Manager, but for some reason create and run job is failing, and there's no reason it should.
                # if st.button("🚀 Create Job", type="primary"):
                #     debug_log("Create Job button clicked")
                #     print("[UI] 🚀 Create Job button clicked")
                #     # Use session state to trigger dialog display
                #     st.session_state.show_job_dialog = True
                #     st.session_state.job_dialog_tables = tables

                # Add test button temporarily for debugging
                if st.button("🔧 Create and Run Job", help="Debug job manager"):
                    st.write("**🔧 Running Job Manager...**")

                    try:
                        if not JOB_MANAGER_AVAILABLE:
                            st.error("❌ Job Manager not available!")
                            st.write(f"Import error: {JOB_MANAGER_IMPORT_ERROR}")
                            return

                        job_manager = DBXMetaGenJobManager(
                            st.session_state.workspace_client
                        )
                        current_user_email = get_current_user_email()
                        job_manager.create_metadata_job(
                            job_name=f"dbxmetagen_test_job_from_app_{current_user_email}",
                            tables=tables,
                            cluster_size="Small (1-2 workers)",
                            config=st.session_state.config,
                            user_email=get_current_user_email(),
                        )
                        st.success("✅ Job Manager created successfully!")
                        st.write(f"- Job Manager type: {type(job_manager)}")
                        st.write(
                            f"- Has create_metadata_job: {hasattr(job_manager, 'create_metadata_job')}"
                        )
                        st.write(
                            f"- Workspace client: {type(st.session_state.workspace_client)}"
                        )

                        # Test with minimal parameters
                        test_tables = ["test.table"] if not tables else tables[:1]
                        st.write(f"- Test tables: {test_tables}")
                        st.write(
                            f"- Config keys: {list(st.session_state.config.keys())}"
                        )

                    except Exception as test_error:
                        st.error(f"❌ Test failed: {str(test_error)}")
                        with st.expander("Error Details"):
                            import traceback

                            st.code(traceback.format_exc())

            with col4:
                if st.button("🗑️ Clear Tables"):
                    st.session_state.selected_tables = []
                    st.rerun()

        else:
            st.warning(
                "⚠️ No tables selected. Please enter table names or upload a CSV file."
            )

        # Check if job dialog should be displayed
        if st.session_state.get("show_job_dialog", False):
            self.show_job_creation_dialog(st.session_state.get("job_dialog_tables", []))
            # Clear the flag after showing
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
        """Create and run a metadata generation job - using working logic from test_job_manager"""
        st.write("**📝 Job Creation Process:**")

        # Validate prerequisites
        if not st.session_state.workspace_client:
            raise RuntimeError("Databricks client not initialized")
        if not tables:
            raise ValueError("No tables specified for processing")
        if not JOB_MANAGER_AVAILABLE:
            raise RuntimeError(f"Job manager not available: {JOB_MANAGER_IMPORT_ERROR}")

        with st.spinner("🚀 Creating and starting job..."):
            # Use the same approach as test_job_manager - load config fresh from YAML
            st.write("• 📁 Loading fresh config from variables.yml...")
        import yaml
        import os

        config_path = "./variables.yml"
        if not os.path.exists(config_path):
            # Try app directory
            config_path = "app/variables.yml"

        with open(config_path, "r") as f:
            fresh_config = yaml.safe_load(f)

        # Generate unique job name like test_job_manager does
        current_user = st.session_state.workspace_client.current_user.me()
        user_email = current_user.user_name
        username = user_email.split("@")[0]  # Get username part before @

        # CRITICAL: Override current_user in config to prevent service principal contamination
        st.write(
            "• 🛠️ Fixing config current_user to prevent service principal contamination..."
        )
        st.write(
            f"• 🔍 Before override: current_user = {fresh_config.get('current_user', 'NOT_FOUND')}"
        )

        # Ensure we set the actual resolved user, not the template
        fresh_config["current_user"] = (
            user_email  # Use actual user, not service principal
        )
        st.write(f"• ✅ After override: current_user = {fresh_config['current_user']}")

        # Also check other config values that might contain the template
        for key, value in fresh_config.items():
            if isinstance(value, str) and "${workspace.current_user.userName}" in value:
                st.write(f"• 🔧 Fixing template in {key}: {value}")
                fresh_config[key] = value.replace(
                    "${workspace.current_user.userName}", user_email
                )
                st.write(f"• ✅ Fixed {key} = {fresh_config[key]}")

        st.write("• 🔧 Initializing job manager...")
        job_manager = DBXMetaGenJobManager(st.session_state.workspace_client)

        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"{username}_app_job_{timestamp}"

        st.write(f"• 📝 Job: {job_name}")
        st.write(f"• 👤 User: {user_email}")

        # Clean up existing jobs with same name (like test_job_manager)
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
                st.write(f"• ⚠️ Warning: Could not delete existing job: {e}")
                st.write("• Continuing with job creation...")
        else:
            st.write("• ✅ No existing job found")

        # Convert tables list to string format like test_job_manager
        if isinstance(tables, list):
            tables_str = ",".join(tables)
        else:
            tables_str = str(tables)

        st.write(f"• 📊 Tables: {tables_str}")
        st.write("• 🚀 Creating and running job...")

        # Create job using fresh config (avoiding service principal contamination)
        job_id, run_id = job_manager.create_metadata_job(
            job_name=job_name,
            tables=tables_str,  # Use string format like test_job_manager
            cluster_size=cluster_size,
            config=fresh_config,  # Use fresh YAML config, not processed session config
            user_email=user_email,
        )

        st.write(f"• ✅ Job created with ID: {job_id}")
        st.write(f"• ✅ Run started with ID: {run_id}")

        # Store in session for monitoring
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
        """Refresh status of all jobs using JobManager (following test_job_manager.py pattern)"""
        if not st.session_state.job_runs:
            return

        # Check if job manager is available
        if not JOB_MANAGER_AVAILABLE:
            st.error("❌ Job manager not available for status updates")
            return

        try:
            job_manager = DBXMetaGenJobManager(st.session_state.workspace_client)

            for run_id, job_info in st.session_state.job_runs.items():
                # Use job manager's get_run_status method (like test_job_manager.py)
                current_status = job_manager.get_run_status(run_id)

                if current_status and current_status != job_info.get("status"):
                    print(
                        f"[JOB_STATUS] Job {run_id} status changed: {job_info.get('status')} → {current_status}"
                    )
                    job_info["status"] = current_status
                    job_info["last_updated"] = datetime.now()

                    # Check if job completed
                    terminal_states = {
                        "SUCCESS",
                        "FAILED",
                        "CANCELED",
                        "TIMEOUT",
                        "TERMINATED",
                    }
                    if current_status in terminal_states:
                        job_info["end_time"] = datetime.now()
                        print(
                            f"[JOB_STATUS] Job {run_id} completed with status: {current_status}"
                        )

        except Exception as e:
            print(f"[JOB_STATUS] Error refreshing status: {str(e)}")
            st.error(f"❌ Error refreshing job status: {str(e)}")

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

    def load_metadata_from_volume(self):
        """Load metadata files from volume for review"""
        if not st.session_state.workspace_client:
            st.warning("⚠️ Databricks client not initialized")
            return

        try:
            with st.spinner("📂 Loading metadata files from volume..."):
                # Get volume path from config
                catalog_name = st.session_state.config.get("catalog_name", "dbxmetagen")
                schema_name = st.session_state.config.get(
                    "schema_name", "metadata_results"
                )
                volume_name = st.session_state.config.get(
                    "volume_name", "generated_metadata"
                )

                volume_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/"

                st.subheader("📁 Select Metadata File from Volume")
                st.info(f"📍 Volume path: {volume_path}")

                # List files in the volume
                try:
                    files = list(
                        st.session_state.workspace_client.files.list_directory_contents(
                            volume_path
                        )
                    )

                    # Filter for metadata files (TSV, CSV)
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

                    if metadata_files:
                        st.success(f"✅ Found {len(metadata_files)} metadata file(s)")

                        # Sort by last modified (newest first)
                        metadata_files.sort(
                            key=lambda x: x["modified"] or 0, reverse=True
                        )

                        # Create a selectbox for file selection
                        file_options = []
                        file_details = {}

                        for file_info in metadata_files:
                            # Format file size
                            size_str = self._format_file_size(file_info["size"])

                            # Format last modified
                            if file_info["modified"]:
                                modified_str = datetime.fromtimestamp(
                                    file_info["modified"] / 1000
                                ).strftime("%Y-%m-%d %H:%M")
                            else:
                                modified_str = "Unknown"

                            display_name = (
                                f"{file_info['name']} ({size_str}, {modified_str})"
                            )
                            file_options.append(display_name)
                            file_details[display_name] = file_info

                        selected_file_display = st.selectbox(
                            "Choose a metadata file:",
                            options=file_options,
                            help="Select a TSV or CSV file containing generated metadata",
                        )

                        if selected_file_display and st.button(
                            "📥 Load Selected File", type="primary"
                        ):
                            selected_file = file_details[selected_file_display]
                            self._load_file_from_volume(selected_file)

                    else:
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

                        # Show alternative upload option
                        st.info(
                            "📤 Alternatively, you can upload a file using the file uploader above."
                        )

                except Exception as file_error:
                    error_msg = str(file_error).lower()
                    if "not found" in error_msg or "does not exist" in error_msg:
                        st.error(f"❌ Volume not found: {volume_path}")
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
                               CREATE VOLUME {catalog_name}.{schema_name}.{volume_name};
                               ```
                            """
                            )
                    else:
                        st.error(f"❌ Error accessing volume: {str(file_error)}")
                        handle_databricks_error(file_error, "volume access")

        except Exception as e:
            st.error(f"❌ Error loading files from volume: {str(e)}")
            handle_databricks_error(e, "volume file listing")

    def _format_file_size(self, size_bytes):
        """Format file size in human-readable format"""
        if size_bytes is None:
            return "Unknown size"

        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def _load_file_from_volume(self, file_info):
        """Load a specific file from volume and process it for review"""
        try:
            with st.spinner(f"📥 Loading {file_info['name']}..."):
                # Read file content using Files API
                file_content = st.session_state.workspace_client.files.download(
                    file_info["path"]
                )

                # Convert bytes to string
                content_str = file_content.contents.decode("utf-8")

                # Create a file-like object for pandas
                from io import StringIO

                file_obj = StringIO(content_str)

                # Determine separator based on file extension
                separator = "\t" if file_info["name"].lower().endswith(".tsv") else ","

                # Read with pandas
                df = pd.read_csv(file_obj, sep=separator)

                st.success(
                    f"✅ Successfully loaded {len(df)} records from {file_info['name']}"
                )

                # Store the loaded data in session state for the review interface
                st.session_state.current_metadata = {
                    "data": df,
                    "filename": file_info["name"],
                    "source": "volume",
                    "path": file_info["path"],
                }

                # Process the loaded data using existing review functionality
                # Create a mock uploaded file object for compatibility
                class MockUploadedFile:
                    def __init__(self, name, dataframe):
                        self.name = name
                        self._df = dataframe

                mock_file = MockUploadedFile(file_info["name"], df)

                # Call the existing review function
                st.markdown("---")
                st.subheader(f"📝 Reviewing: {file_info['name']}")
                self._review_loaded_metadata(df, file_info["name"])

        except Exception as e:
            st.error(f"❌ Error loading file {file_info['name']}: {str(e)}")
            handle_databricks_error(e, "file loading")

    def _review_loaded_metadata(self, df, filename):
        """Review loaded metadata (extracted from review_uploaded_metadata for reuse)"""
        try:
            # Show data info
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
                        st.metric(
                            "Processing Mode",
                            df["mode"].iloc[0] if not df.empty else "Unknown",
                        )

            # Filter section
            st.subheader("🔍 Filter & Select")
            col1, col2 = st.columns(2)

            with col1:
                if "table_name" in df.columns:
                    selected_tables = st.multiselect(
                        "Filter by Tables",
                        options=sorted(df["table_name"].unique()),
                        default=(
                            sorted(df["table_name"].unique())[:5]
                            if len(df["table_name"].unique()) > 5
                            else sorted(df["table_name"].unique())
                        ),
                        key="volume_table_filter",
                    )
                    if selected_tables:
                        df = df[df["table_name"].isin(selected_tables)]

            with col2:
                if "column_name" in df.columns:
                    search_term = st.text_input(
                        "Search Columns",
                        placeholder="Enter column name or pattern...",
                        help="Filter columns by name",
                        key="volume_column_search",
                    )
                    if search_term:
                        df = df[
                            df["column_name"].str.contains(
                                search_term, case=False, na=False
                            )
                        ]

            # Editable data section
            st.subheader("✏️ Edit Metadata")
            st.info(
                "💡 Double-click on cells to edit. Focus on comment and tag columns."
            )

            # Configure editable columns
            if not df.empty:
                # Use st.data_editor for inline editing
                edited_df = st.data_editor(
                    df,
                    use_container_width=True,
                    height=400,
                    column_config={
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
                    },
                    disabled=(
                        ["table_name", "column_name", "data_type"]
                        if "data_type" in df.columns
                        else ["table_name", "column_name"]
                    ),
                    key="volume_metadata_editor",
                )

                # Action buttons
                st.subheader("🚀 Actions")
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button(
                        "💾 Save Edited Metadata", type="primary", key="volume_save"
                    ):
                        self.save_edited_metadata(edited_df, filename)

                with col2:
                    if st.button("🔄 Generate DDL", key="volume_generate_ddl"):
                        self.generate_ddl_from_metadata(edited_df)

                with col3:
                    if st.button(
                        "✅ Apply to Tables", type="secondary", key="volume_apply"
                    ):
                        self.apply_metadata_to_tables(edited_df)

        except Exception as e:
            st.error(f"❌ Error processing metadata: {str(e)}")

    def review_uploaded_metadata(self, uploaded_file):
        """Process and allow editing of uploaded metadata"""
        try:
            # Determine separator
            file_ext = uploaded_file.name.split(".")[-1].lower()
            separator = "\t" if file_ext == "tsv" else ","

            # Load the metadata
            df = pd.read_csv(uploaded_file, sep=separator)

            st.success(
                f"✅ Loaded {len(df)} metadata records from {uploaded_file.name}"
            )

            # Show data info
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
                        st.metric(
                            "Processing Mode",
                            df["mode"].iloc[0] if not df.empty else "Unknown",
                        )

            # Filter section
            st.subheader("🔍 Filter & Select")
            col1, col2 = st.columns(2)

            with col1:
                if "table_name" in df.columns:
                    selected_tables = st.multiselect(
                        "Filter by Tables",
                        options=sorted(df["table_name"].unique()),
                        default=(
                            sorted(df["table_name"].unique())[:5]
                            if len(df["table_name"].unique()) > 5
                            else sorted(df["table_name"].unique())
                        ),
                    )
                    if selected_tables:
                        df = df[df["table_name"].isin(selected_tables)]

            with col2:
                if "column_name" in df.columns:
                    search_term = st.text_input(
                        "Search Columns",
                        placeholder="Enter column name or pattern...",
                        help="Filter columns by name",
                    )
                    if search_term:
                        df = df[
                            df["column_name"].str.contains(
                                search_term, case=False, na=False
                            )
                        ]

            # Editable data section
            st.subheader("✏️ Edit Metadata")
            st.info(
                "💡 Double-click on cells to edit. Focus on comment and tag columns."
            )

            # Configure editable columns
            if not df.empty:
                # Use st.data_editor for inline editing
                edited_df = st.data_editor(
                    df,
                    use_container_width=True,
                    height=400,
                    column_config={
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
                    },
                    disabled=(
                        ["table_name", "column_name", "data_type"]
                        if "data_type" in df.columns
                        else ["table_name", "column_name"]
                    ),
                    key="metadata_editor",
                )

                # Action buttons
                st.subheader("🚀 Actions")
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("💾 Save Edited Metadata", type="primary"):
                        self.save_edited_metadata(edited_df, uploaded_file.name)

                with col2:
                    if st.button("🔄 Generate DDL"):
                        self.generate_ddl_from_metadata(edited_df)

                with col3:
                    if st.button("✅ Apply to Tables", type="secondary"):
                        self.apply_metadata_to_tables(edited_df)

        except Exception as e:
            st.error(f"❌ Error processing metadata file: {str(e)}")

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

    def apply_metadata_to_tables(self, df: pd.DataFrame):
        """Apply metadata directly to tables using sync_reviewed_ddl notebook"""
        if not st.session_state.workspace_client:
            st.warning("⚠️ Databricks client not initialized")
            return

        try:
            with st.spinner("🚀 Applying metadata to tables..."):
                # Prepare parameters for sync_reviewed_ddl notebook
                catalog_name = st.session_state.config.get("catalog_name", "dbxmetagen")
                schema_name = st.session_state.config.get(
                    "schema_name", "metadata_results"
                )
                volume_name = st.session_state.config.get(
                    "volume_name", "generated_metadata"
                )

                # Save the edited metadata to a temporary file name that the notebook can pick up
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                reviewed_filename = f"reviewed_metadata_{timestamp}.tsv"

                # Convert DataFrame to TSV format for the notebook
                tsv_data = df.to_csv(sep="\t", index=False)

                # Show preview of what will be applied
                st.subheader("📋 Preview: Changes to Apply")
                st.info(
                    f"Will apply metadata from {len(df)} records to {df['table_name'].nunique()} tables"
                )

                # In a full implementation, you would:
                # 1. Upload the reviewed TSV to the volume
                # 2. Run the sync_reviewed_ddl notebook with the file path
                # 3. Monitor the job execution

                st.warning("🚧 **Manual Step Required:**")
                st.markdown(
                    f"""
                To apply the reviewed metadata:
                1. **Download the reviewed metadata** using the "Save Edited Metadata" button above
                2. **Upload it to your volume** at: `/Volumes/{catalog_name}/{schema_name}/{volume_name}/`
                3. **Run the sync_reviewed_ddl notebook** with the uploaded file path
                
                Or create a job to automate this process using the sync_reviewed_ddl notebook.
                """
                )

                # Provide the option to create a job for this
                if st.button("🚀 Create Sync Job"):
                    self.create_sync_metadata_job(df, reviewed_filename)

        except Exception as e:
            st.error(f"❌ Error applying metadata: {str(e)}")

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
