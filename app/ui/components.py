"""
UI components module.
Contains all Streamlit rendering functions and UI components.
"""

import streamlit as st
import pandas as pd
import yaml
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import base64

from core.config import ConfigManager, DatabricksClientManager
from core.jobs import JobManager
from core.data_ops import DataOperations, MetadataProcessor

logger = logging.getLogger(__name__)


class UIComponents:
    """Contains all UI rendering methods."""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.data_ops = DataOperations()
        self.metadata_processor = MetadataProcessor()
        # Job manager will be initialized when workspace client is available
        self._job_manager = None

    @property
    def job_manager(self):
        """Get job manager, initializing if needed"""
        if self._job_manager is None and st.session_state.get("workspace_client"):
            self._job_manager = JobManager(st.session_state.workspace_client)
        return self._job_manager

    def render_sidebar_config(self):
        """Render configuration sidebar."""
        st.sidebar.header("⚙️ Configuration")

        # Load default configuration
        if not st.session_state.config:
            st.session_state.config = self.config_manager.load_default_config()

        # with st.sidebar.expander("📂 Configuration File", expanded=False):
        #     col1, col2 = st.sidebar.columns(2)

        #     with col1:
        #         if st.button("💾 Save Config"):
        #             self._save_config_to_file()

        #     with col2:
        #         uploaded_config = st.file_uploader(
        #             "📤 Load Config", type=["yml", "yaml"], key="config_uploader"
        #         )
        #         if uploaded_config:
        #             self._load_config_from_file(uploaded_config)

        # Configuration form
        with st.sidebar.form("config_form"):
            st.subheader("🏷️ Target Settings")

            catalog_name = st.text_input(
                "Catalog Name",
                value=st.session_state.config.get("catalog_name", "dbxmetagen"),
                help="Unity Catalog name for storing results",
            )

            schema_name = st.text_input(
                "Schema Name",
                value=st.session_state.config.get("schema_name", "metadata_results"),
                help="Schema name for metadata tables",
            )

            st.subheader("⚡ Processing Options")

            allow_data = st.checkbox(
                "Allow Data Sampling",
                value=st.session_state.config.get("allow_data", True),
                help="Include actual data samples in LLM processing",
            )

            sample_size = st.number_input(
                "Sample Size",
                min_value=0,
                max_value=1000,
                value=st.session_state.config.get("sample_size", 100),
                help="Number of rows to sample per column (0 = no sampling)",
            )

            mode = st.selectbox(
                "Processing Mode",
                options=["both", "comments", "pii"],
                index=["both", "comments", "pii"].index(
                    st.session_state.config.get("mode", "both")
                ),
                help="What to generate: comments, PII classification, or both",
            )

            st.subheader("🚀 Execution Settings")

            cluster_size = st.selectbox(
                "Cluster Size",
                options=["small", "medium", "large"],
                index=["small", "medium", "large"].index(
                    st.session_state.config.get("cluster_size", "small")
                ),
            )

            apply_metadata = st.checkbox(
                "⚠️ Apply DDL (DANGEROUS)",
                value=st.session_state.config.get("apply_metadata", False),
                help="WARNING: This will directly modify your tables!",
            )

            if st.form_submit_button("💾 Save Configuration", type="primary"):
                # Update session state config
                st.session_state.config.update(
                    {
                        "catalog_name": catalog_name,
                        "schema_name": schema_name,
                        "allow_data": allow_data,
                        "sample_size": sample_size,
                        "mode": mode,
                        "cluster_size": cluster_size,
                        "apply_metadata": apply_metadata,
                    }
                )

                st.sidebar.success("✅ Configuration saved!")

    def render_unified_table_management(self):
        """Render the unified table management interface."""
        st.header("📋 Table Management")

        tables = self._render_table_input_section()

        if tables:
            self._render_table_action_buttons(tables)
        else:
            self._show_no_tables_warning()

    def _render_table_input_section(self) -> List[str]:
        """Render table input section and return list of tables."""
        col1, col2 = st.columns([2, 2])

        with col1:
            st.subheader("Enter Table Names")
            table_names_input = st.text_area(
                "Table Names (one per line)",
                value="\n".join(st.session_state.get("selected_tables", [])),
                height=150,
                help="Enter fully qualified table names: catalog.schema.table",
            )

        with col2:
            st.subheader("Or Upload CSV")
            uploaded_file = st.file_uploader(
                "Upload CSV File",
                type=["csv"],
                help="CSV file with 'table_name' column",
            )

            if uploaded_file:
                csv_tables = self.data_ops.process_uploaded_csv(uploaded_file)
                if csv_tables:
                    # Update the text area
                    st.session_state.selected_tables = csv_tables
                    st.rerun()

        # Parse and validate tables
        tables = self._parse_and_store_tables(table_names_input)

        return tables

    def _parse_and_store_tables(self, table_names_input: str) -> List[str]:
        """Parse table names and store in session state."""
        tables = [
            name.strip() for name in table_names_input.split("\n") if name.strip()
        ]
        st.session_state.selected_tables = tables
        return tables

    def _render_table_action_buttons(self, tables: List[str]):
        """Render action buttons for table operations."""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("✅ Validate Tables", type="secondary"):
                self._validate_tables(tables)

        with col2:
            if st.button("💾 Save List", type="secondary"):
                self._save_table_list(tables)

        with col3:
            self._render_job_creation_button(tables)

        with col4:
            self._debug_job_manager_creation(tables)

    def _render_job_creation_button(self, tables: List[str]):
        """Render job creation button with working dialog trigger"""
        if st.button("🚀 Create & Run Job", type="primary"):
            # Show the job dialog directly (working implementation)
            self.show_job_creation_dialog(tables)

    def _debug_job_manager_creation(self, tables: List[str]):
        """Debug job manager creation - using working implementation"""
        if st.button("🔧 Debug Job Creation"):
            if not self.job_manager:
                st.error(
                    "❌ Databricks client not initialized. Please check connection."
                )
                return
            with st.spinner("Creating debug job..."):
                self.job_manager.debug_job_manager_creation(
                    tables[:3]
                )  # Limit to 3 tables for testing

    def _show_no_tables_warning(self):
        """Show warning when no tables are provided."""
        st.info("📝 Enter table names above or upload a CSV file to get started")

    def _validate_tables(self, tables: List[str]):
        """Validate table names and accessibility."""
        with st.spinner("Validating tables..."):
            # Format validation
            valid_tables, invalid_tables = self.data_ops.validate_table_names(tables)
            self.data_ops.display_table_validation_results(valid_tables, invalid_tables)

            # Accessibility validation for valid tables
            if valid_tables:
                st.subheader("🔍 Checking Table Access...")
                validation_results = self.data_ops.validate_tables(valid_tables)

                accessible = validation_results["accessible"]
                inaccessible = validation_results["inaccessible"]
                errors = validation_results["errors"]

                if accessible:
                    st.success(f"✅ {len(accessible)} tables accessible")
                    with st.expander("Accessible Tables"):
                        for table in accessible:
                            st.write(f"✅ {table}")

                if inaccessible:
                    st.error(f"❌ {len(inaccessible)} tables inaccessible")
                    with st.expander("Inaccessible Tables"):
                        for table in inaccessible:
                            st.write(f"❌ {table}")

                if errors:
                    st.error("Validation Errors:")
                    for error in errors:
                        st.write(f"⚠️ {error}")

    def _save_table_list(self, tables: List[str]):
        """Save table list as downloadable CSV."""
        csv_content = self.data_ops.save_table_list(tables)
        if csv_content:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"table_names_{timestamp}.csv"

            st.download_button(
                label="📥 Download CSV",
                data=csv_content,
                file_name=filename,
                mime="text/csv",
            )

    def show_job_creation_dialog(self, tables: List[str]):
        """Show job creation dialog with configuration options (WORKING IMPLEMENTATION)"""
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
                    "Allow Data": st.session_state.config.get("allow_data", False),
                    "Sample Size": st.session_state.config.get("sample_size", 5),
                    "Apply DDL": st.session_state.config.get("apply_ddl", False),
                }

                for key, value in config_preview.items():
                    st.write(f"**{key}:** {value}")

            # Create and run button (WORKING IMPLEMENTATION)
            if st.button("🚀 Create & Run Job", type="primary", key="create_job_main"):
                logger.info(
                    f"Create & Run Job button clicked - job_name: {job_name}, tables: {len(tables)}, cluster_size: {cluster_size}"
                )

                if not self.job_manager:
                    st.error(
                        "❌ Databricks client not initialized. Please check connection."
                    )
                    return

                try:
                    self.job_manager.create_and_run_job(job_name, tables, cluster_size)
                except Exception as create_job_e:
                    st.error(f"❌ Failed to create job: {create_job_e}")
                    logger.error(f"ERROR in create_and_run_job: {create_job_e}")
                    import traceback

                    st.code(traceback.format_exc())

    def render_job_status_section(self):
        """Render job status monitoring section."""
        st.header("📊 Job Status")

        if not st.session_state.get("job_runs"):
            st.info("No jobs to display. Create a job above to track its status.")
            return

        # Job status header with manual refresh button (WORKING IMPLEMENTATION)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("📊 Job Status")
        with col2:
            if st.button("🔄 Refresh Now", help="Manually refresh job status"):
                if not self.job_manager:
                    st.error(
                        "❌ Databricks client not initialized. Please check connection."
                    )
                    return
                with st.spinner("Refreshing job status..."):
                    self.job_manager.refresh_job_status()
                st.success("✅ Status refreshed!")
                st.rerun()

        # Display job status using working format (run_id as key)
        self._display_job_status_working()

    def _display_job_status_working(self):
        """Display job status using working format (run_id as key)"""
        for run_id, job_info in st.session_state.job_runs.items():
            # Calculate progress for running jobs
            progress = 0
            if job_info["status"] == "SUCCESS":
                progress = 100
            elif job_info["status"] == "RUNNING":
                # Estimate progress based on time elapsed (rough estimate)
                start_time = job_info.get("start_time", datetime.now())
                if isinstance(start_time, str):
                    # Handle string timestamps
                    try:
                        start_time = datetime.fromisoformat(start_time)
                    except:
                        start_time = datetime.now()

                elapsed = (datetime.now() - start_time).total_seconds()
                progress = min(80, (elapsed / 600) * 100)  # Max 80% for running jobs
            elif job_info["status"] == "FAILED":
                progress = 0

            with st.expander(
                f"🔧 {job_info['job_name']} (Run ID: {run_id})",
                expanded=job_info["status"] == "RUNNING",
            ):
                # Job details
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Job ID:** {job_info.get('job_id', 'Unknown')}")
                    st.write(f"**Status:** {job_info['status']}")
                    st.write(f"**Tables:** {len(job_info.get('tables', []))}")

                with col2:
                    st.write(
                        f"**Cluster Size:** {job_info.get('cluster_size', 'Unknown')}"
                    )
                    st.write(f"**Created:** {job_info.get('created_at', 'Unknown')}")
                    if job_info.get("result_state"):
                        st.write(f"**Result:** {job_info['result_state']}")

                # Progress bar
                st.progress(progress / 100)

                # Table list
                tables = job_info.get("tables", [])
                if tables:
                    with st.expander(f"📋 Processing {len(tables)} Tables"):
                        for i, table in enumerate(tables[:10], 1):
                            st.write(f"{i}. {table}")
                        if len(tables) > 10:
                            st.write(f"... and {len(tables) - 10} more tables")

    def _display_job_status(self):
        """Display current job status."""
        for job_id, job_info in st.session_state.job_runs.items():
            with st.container():
                # Job header
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.subheader(f"Job {job_id}: {job_info.get('job_name', 'Unknown')}")

                with col2:
                    status = job_info.get("status", "UNKNOWN")
                    if status == "RUNNING":
                        st.info(f"🔄 {status}")
                    elif status == "SUCCESS":
                        st.success(f"✅ {status}")
                    elif status in ["FAILED", "CANCELLED"]:
                        st.error(f"❌ {status}")
                    else:
                        st.warning(f"⚠️ {status}")

                with col3:
                    created_at = job_info.get("created_at", "Unknown")
                    if isinstance(created_at, str):
                        try:
                            created = datetime.fromisoformat(created_at)
                            st.write(f"Created: {created.strftime('%H:%M')}")
                        except:
                            st.write(f"Created: {created_at}")

                # Job details
                with st.expander("Job Details"):
                    st.write(f"**Run ID:** {job_info.get('run_id', 'Unknown')}")
                    st.write(
                        f"**Cluster Size:** {job_info.get('cluster_size', 'Unknown')}"
                    )
                    st.write(f"**Tables:** {len(job_info.get('tables', []))}")

                    # Show table list
                    tables = job_info.get("tables", [])
                    if tables:
                        st.write("**Table List:**")
                        for table in tables[:10]:  # Show first 10
                            st.write(f"• {table}")
                        if len(tables) > 10:
                            st.write(f"... and {len(tables) - 10} more tables")

    def render_results_viewer(self):
        """Render results viewing interface."""
        st.header("📊 Results Viewer")

        # Configuration inputs
        col1, col2, col3 = st.columns(3)

        with col1:
            catalog = st.text_input(
                "Catalog",
                value=st.session_state.config.get("catalog_name", "dbxmetagen"),
            )

        with col2:
            schema = st.text_input(
                "Schema",
                value=st.session_state.config.get("schema_name", "metadata_results"),
            )

        with col3:
            volume = st.text_input(
                "Volume",
                value=st.session_state.config.get("volume_name", "generated_metadata"),
            )

        if st.button("📥 Load Results"):
            with st.spinner("Loading results from volume..."):
                df = self.metadata_processor.load_metadata_from_volume(
                    catalog, schema, volume
                )
                if df is not None:
                    st.session_state.current_metadata = df

        # Display results
        if st.session_state.get("current_metadata") is not None:
            df = st.session_state.current_metadata

            st.success(f"✅ Loaded {len(df)} metadata records")

            # Display options
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📥 Download TSV"):
                    self._download_metadata(df, "tsv")

            with col2:
                if st.button("📥 Download CSV"):
                    self._download_metadata(df, "csv")

            # Display data
            st.dataframe(df, use_container_width=True)

    def render_metadata_review(self):
        """Render metadata review interface."""
        st.header("✏️ Review Metadata")

        uploaded_file = st.file_uploader(
            "Upload Metadata File for Review",
            type=["tsv", "csv"],
            help="Upload a TSV or CSV file with metadata to review",
        )

        if uploaded_file:
            df = self.metadata_processor.review_uploaded_metadata(uploaded_file)
            if df is not None:
                st.subheader("📋 Metadata Review")
                st.dataframe(df, use_container_width=True)

                st.subheader("🚀 Apply Changes")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("✅ Apply to Tables"):
                        self._apply_metadata(df)

                with col2:
                    if st.button("🚀 Create Sync Job"):
                        if not self.job_manager:
                            st.error(
                                "❌ Databricks client not initialized. Please check connection."
                            )
                        else:
                            self.job_manager.create_sync_metadata_job(
                                df, uploaded_file.name
                            )

    def _apply_metadata(self, df: pd.DataFrame):
        """Apply metadata changes to tables."""
        with st.spinner("Applying metadata to tables..."):
            results = self.metadata_processor.apply_metadata_to_tables(df)

            if results["success"]:
                st.success(f"✅ Applied metadata to {results['applied']} tables")
            else:
                st.error(
                    f"❌ Failed to apply metadata: {results.get('error', 'Unknown error')}"
                )

                if results.get("errors"):
                    with st.expander("Error Details"):
                        for error in results["errors"]:
                            st.write(f"• {error}")

    def _download_metadata(self, df: pd.DataFrame, format: str):
        """Download metadata in specified format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "tsv":
            content = df.to_csv(sep="\t", index=False)
            filename = f"metadata_{timestamp}.tsv"
            mime = "text/tab-separated-values"
        else:
            content = df.to_csv(index=False)
            filename = f"metadata_{timestamp}.csv"
            mime = "text/csv"

        st.download_button(
            label=f"📥 Download {format.upper()}",
            data=content.encode("utf-8"),
            file_name=filename,
            mime=mime,
        )

    def _save_config_to_file(self):
        """Save current configuration to YAML file."""
        try:
            yaml_content = yaml.dump(st.session_state.config, default_flow_style=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dbxmetagen_config_{timestamp}.yml"

            st.sidebar.download_button(
                label="📥 Download Config",
                data=yaml_content.encode("utf-8"),
                file_name=filename,
                mime="application/x-yaml",
            )

        except Exception as e:
            st.sidebar.error(f"❌ Failed to save config: {str(e)}")

    def _load_config_from_file(self, uploaded_file):
        """Load configuration from uploaded YAML file."""
        try:
            content = uploaded_file.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8")

            config = yaml.safe_load(content)
            st.session_state.config.update(config)

            st.sidebar.success("✅ Configuration loaded!")
            st.rerun()

        except Exception as e:
            st.sidebar.error(f"❌ Failed to load config: {str(e)}")

    def render_help(self):
        """Render help and documentation."""
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


# Job dialog is now handled directly in the button click (working implementation)
def handle_job_dialog_display():
    """DEPRECATED - Job dialog now handled directly in button click."""
    pass
