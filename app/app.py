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
from databricks.sdk.service.jobs import RunNow, JobSettings, NotebookTask, JobCluster, ClusterSpec
import base64

# Configure page
st.set_page_config(
    page_title="DBX MetaGen",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "config" not in st.session_state:
    st.session_state.config = {}
if "job_runs" not in st.session_state:
    st.session_state.job_runs = {}
if "workspace_client" not in st.session_state:
    st.session_state.workspace_client = None

class DBXMetaGenApp:
    def __init__(self):
        self.setup_client()
        
    def setup_client(self):
        """Initialize Databricks workspace client"""
        try:
            # Try to get token from environment or Databricks context
            if hasattr(st, 'secrets') and 'databricks' in st.secrets:
                token = st.secrets.databricks.get('token')
                host = st.secrets.databricks.get('host')
            else:
                # Fallback to environment variables
                token = os.environ.get('DATABRICKS_TOKEN')
                host = os.environ.get('DATABRICKS_HOST')
            
            if token and host:
                st.session_state.workspace_client = WorkspaceClient(
                    host=host,
                    token=token
                )
            else:
                st.warning("⚠️ Databricks credentials not found. Some features may not work.")
        except Exception as e:
            st.error(f"❌ Failed to initialize Databricks client: {str(e)}")

    def load_default_config(self) -> Dict[str, Any]:
        """Load default configuration from variables.yml"""
        default_config = {
            "catalog_name": "dbxmetagen",
            "host": "https://your-databricks-workspace.cloud.databricks.com/",
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
            "volume_name": "generated_metadata"
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
                value=config.get("catalog_name", "dbxmetagen"),
                help="Target catalog where data, models, and files are stored"
            )
            
            config["host"] = st.text_input(
                "Databricks Host", 
                value=config.get("host", ""),
                help="Your Databricks workspace URL"
            )
            
            config["schema_name"] = st.text_input(
                "Schema Name",
                value=config.get("schema_name", "metadata_results"),
                help="Schema where results will be stored"
            )
        
        # Data Settings
        with st.sidebar.expander("📊 Data Settings"):
            config["allow_data"] = st.checkbox(
                "Allow Data in Processing", 
                value=config.get("allow_data", True),
                help="Set to false to prevent data from being sent to LLMs"
            )
            
            config["sample_size"] = st.number_input(
                "Sample Size", 
                min_value=0, 
                max_value=50, 
                value=config.get("sample_size", 5),
                help="Number of data samples per column for analysis"
            )
            
            config["mode"] = st.selectbox(
                "Processing Mode",
                options=["comment", "pi", "both"],
                index=["comment", "pi", "both"].index(config.get("mode", "comment")),
                help="Mode of operation: generate comments, identify PII, or both"
            )
        
        # Advanced Settings
        with st.sidebar.expander("🔧 Advanced Settings"):
            config["model"] = st.selectbox(
                "LLM Model",
                options=[
                    "databricks-claude-3-7-sonnet",
                    "databricks-meta-llama-3-3-70b-instruct",
                    "databricks-claude-3-5-sonnet"
                ],
                index=0,
                help="LLM model for metadata generation"
            )
            
            config["apply_ddl"] = st.checkbox(
                "Apply DDL Directly",
                value=config.get("apply_ddl", False),
                help="⚠️ WARNING: This will modify your tables directly"
            )
            
            config["temperature"] = st.slider(
                "Model Temperature",
                min_value=0.0,
                max_value=1.0,
                value=config.get("temperature", 0.1),
                step=0.1,
                help="Model creativity level (0.0 = deterministic, 1.0 = creative)"
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
                "📁 Load Config",
                type=['yml', 'yaml'],
                key="config_upload"
            )
            if uploaded_config:
                self.load_config_from_file(uploaded_config)

    def save_config_to_file(self):
        """Save current configuration to downloadable file"""
        config_yaml = yaml.dump({"variables": {
            k: {"default": v} for k, v in st.session_state.config.items()
        }}, default_flow_style=False)
        
        st.sidebar.download_button(
            label="⬇️ Download Config",
            data=config_yaml,
            file_name=f"dbxmetagen_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yml",
            mime="text/yaml"
        )

    def load_config_from_file(self, uploaded_file):
        """Load configuration from uploaded file"""
        try:
            config_data = yaml.safe_load(uploaded_file)
            if "variables" in config_data:
                new_config = {k: v.get("default", v) for k, v in config_data["variables"].items()}
                st.session_state.config.update(new_config)
                st.sidebar.success("✅ Configuration loaded successfully!")
                st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Error loading config: {str(e)}")

    def render_table_management(self):
        """Render table management section"""
        st.header("📋 Table Management")
        
        tab1, tab2 = st.tabs(["📝 Table Selection", "📤 Upload CSV"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                table_names_input = st.text_area(
                    "Table Names (one per line)",
                    height=200,
                    placeholder="catalog.schema.table1\ncatalog.schema.table2\n...",
                    help="Enter fully qualified table names, one per line"
                )
            
            with col2:
                if st.button("🔍 Validate Tables", type="secondary"):
                    if table_names_input.strip():
                        tables = [t.strip() for t in table_names_input.split('\n') if t.strip()]
                        self.validate_tables(tables)
                
                if st.button("💾 Save Table List"):
                    if table_names_input.strip():
                        tables = [t.strip() for t in table_names_input.split('\n') if t.strip()]
                        self.save_table_list(tables)
        
        with tab2:
            uploaded_file = st.file_uploader(
                "Upload table_names.csv",
                type=['csv'],
                help="CSV file with 'table_name' column containing fully qualified table names"
            )
            
            if uploaded_file:
                self.process_uploaded_csv(uploaded_file)

    def validate_tables(self, tables: List[str]):
        """Validate that tables exist and are accessible"""
        if not st.session_state.workspace_client:
            st.warning("⚠️ Cannot validate tables: Databricks client not initialized")
            return
        
        with st.spinner("🔍 Validating tables..."):
            valid_tables = []
            invalid_tables = []
            
            for table in tables:
                try:
                    # Simple validation - try to get table info
                    parts = table.split('.')
                    if len(parts) == 3:
                        valid_tables.append(table)
                    else:
                        invalid_tables.append(f"{table} (invalid format)")
                except Exception as e:
                    invalid_tables.append(f"{table} ({str(e)})")
            
            if valid_tables:
                st.success(f"✅ {len(valid_tables)} valid tables found")
                with st.expander("Valid Tables"):
                    for table in valid_tables:
                        st.write(f"• {table}")
            
            if invalid_tables:
                st.error(f"❌ {len(invalid_tables)} invalid tables found")
                with st.expander("Invalid Tables"):
                    for table in invalid_tables:
                        st.write(f"• {table}")

    def save_table_list(self, tables: List[str]):
        """Save table list as CSV for download"""
        df = pd.DataFrame({"table_name": tables})
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="⬇️ Download table_names.csv",
            data=csv,
            file_name=f"table_names_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    def process_uploaded_csv(self, uploaded_file):
        """Process uploaded CSV file"""
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'table_name' not in df.columns:
                st.error("❌ CSV must contain a 'table_name' column")
                return
            
            tables = df['table_name'].dropna().tolist()
            
            st.success(f"✅ Loaded {len(tables)} tables from CSV")
            
            # Display preview
            with st.expander("📋 Table Preview"):
                st.dataframe(df.head(10))
            
            # Store in session state for job creation
            st.session_state.selected_tables = tables
            
        except Exception as e:
            st.error(f"❌ Error processing CSV: {str(e)}")

    def render_job_management(self):
        """Render job management section"""
        st.header("🚀 Job Management")
        
        if not st.session_state.workspace_client:
            st.warning("⚠️ Databricks client not initialized. Cannot manage jobs.")
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📋 Job Configuration")
            
            job_name = st.text_input(
                "Job Name",
                value=f"dbxmetagen_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                help="Name for the metadata generation job"
            )
            
            cluster_size = st.selectbox(
                "Cluster Size",
                options=["Small (1-2 workers)", "Medium (2-4 workers)", "Large (4-8 workers)"],
                index=1,
                help="Cluster size for the job"
            )
            
            # Get tables from session state or input
            tables_source = st.radio(
                "Table Source",
                options=["Manual Input", "Uploaded CSV"],
                help="Source of table names for processing"
            )
            
            if tables_source == "Manual Input":
                table_names_for_job = st.text_area(
                    "Tables for Job",
                    height=100,
                    placeholder="catalog.schema.table1,catalog.schema.table2,...",
                    help="Comma-separated list of table names"
                )
                tables_list = [t.strip() for t in table_names_for_job.split(',') if t.strip()] if table_names_for_job else []
            else:
                tables_list = st.session_state.get('selected_tables', [])
                if tables_list:
                    st.info(f"📋 Using {len(tables_list)} tables from uploaded CSV")
                else:
                    st.warning("⚠️ No tables loaded from CSV. Please upload a CSV file first.")
        
        with col2:
            st.subheader("🎯 Job Actions")
            
            if st.button("🚀 Create & Run Job", type="primary", disabled=not tables_list):
                self.create_and_run_job(job_name, tables_list, cluster_size)
            
            if st.button("📊 Refresh Job Status"):
                self.refresh_job_status()
        
        # Display active jobs
        self.display_job_status()

    def create_and_run_job(self, job_name: str, tables: List[str], cluster_size: str):
        """Create and run a metadata generation job"""
        try:
            with st.spinner("🚀 Creating and starting job..."):
                # Map cluster size to worker count
                worker_map = {
                    "Small (1-2 workers)": {"min": 1, "max": 2},
                    "Medium (2-4 workers)": {"min": 2, "max": 4},
                    "Large (4-8 workers)": {"min": 4, "max": 8}
                }
                workers = worker_map[cluster_size]
                
                # Create job configuration
                job_config = JobSettings(
                    name=job_name,
                    tasks=[
                        {
                            "task_key": "generate_metadata",
                            "notebook_task": NotebookTask(
                                notebook_path="./notebooks/generate_metadata",
                                base_parameters={
                                    "table_names": ",".join(tables),
                                    "mode": st.session_state.config.get("mode", "comment"),
                                    "env": "app"
                                }
                            ),
                            "job_cluster_key": "metadata_cluster"
                        }
                    ],
                    job_clusters=[
                        JobCluster(
                            job_cluster_key="metadata_cluster",
                            new_cluster=ClusterSpec(
                                spark_version="16.4.x-cpu-ml-scala2.12",
                                node_type_id="Standard_D3_v2",  # Adjust based on cloud provider
                                autoscale={
                                    "min_workers": workers["min"],
                                    "max_workers": workers["max"]
                                }
                            )
                        )
                    ],
                    email_notifications={
                        "on_failure": [st.session_state.workspace_client.current_user.me().user_name]
                    }
                )
                
                # Create the job
                job = st.session_state.workspace_client.jobs.create(**job_config)
                
                # Run the job
                run = st.session_state.workspace_client.jobs.run_now(
                    job_id=job.job_id,
                    notebook_params={
                        "table_names": ",".join(tables),
                        "mode": st.session_state.config.get("mode", "comment")
                    }
                )
                
                # Store job info
                st.session_state.job_runs[run.run_id] = {
                    "job_id": job.job_id,
                    "job_name": job_name,
                    "run_id": run.run_id,
                    "tables": tables,
                    "status": "RUNNING",
                    "start_time": datetime.now(),
                    "mode": st.session_state.config.get("mode", "comment")
                }
                
                st.success(f"✅ Job created and started! Job ID: {job.job_id}, Run ID: {run.run_id}")
                
        except Exception as e:
            st.error(f"❌ Error creating job: {str(e)}")

    def refresh_job_status(self):
        """Refresh status of all jobs"""
        if not st.session_state.job_runs:
            st.info("ℹ️ No active jobs to refresh")
            return
        
        try:
            with st.spinner("🔄 Refreshing job status..."):
                for run_id, job_info in st.session_state.job_runs.items():
                    run_details = st.session_state.workspace_client.jobs.get_run(run_id)
                    job_info["status"] = run_details.state.life_cycle_state.value
                    
                    if run_details.end_time:
                        job_info["end_time"] = datetime.fromtimestamp(run_details.end_time / 1000)
            
            st.success("✅ Job status refreshed")
            
        except Exception as e:
            st.error(f"❌ Error refreshing job status: {str(e)}")

    def display_job_status(self):
        """Display current job status"""
        if not st.session_state.job_runs:
            st.info("ℹ️ No jobs running")
            return
        
        st.subheader("📊 Job Status")
        
        for run_id, job_info in st.session_state.job_runs.items():
            with st.expander(f"🔧 {job_info['job_name']} (Run ID: {run_id})"):
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    status_color = {
                        "RUNNING": "🟡",
                        "SUCCESS": "🟢", 
                        "FAILED": "🔴",
                        "TERMINATED": "🟠"
                    }.get(job_info["status"], "⚪")
                    
                    st.write(f"**Status:** {status_color} {job_info['status']}")
                    st.write(f"**Mode:** {job_info['mode']}")
                    st.write(f"**Tables:** {len(job_info['tables'])}")
                
                with col2:
                    st.write(f"**Started:** {job_info['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                    if "end_time" in job_info:
                        st.write(f"**Ended:** {job_info['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                with col3:
                    if st.button(f"🔗 View in Databricks", key=f"view_{run_id}"):
                        databricks_url = f"{st.session_state.config['host']}#job/{job_info['job_id']}/run/{run_id}"
                        st.write(f"[Open Job Run]({databricks_url})")

    def render_results_viewer(self):
        """Render results viewing section"""
        st.header("📊 Results Viewer")
        
        tab1, tab2, tab3 = st.tabs(["📄 Generated Files", "🏷️ Table Metadata", "📈 Statistics"])
        
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
            type=['tsv', 'sql', 'csv'],
            help="Upload TSV, SQL, or CSV files generated by dbxmetagen"
        )
        
        if uploaded_result:
            file_type = uploaded_result.name.split('.')[-1].lower()
            
            try:
                if file_type in ['tsv', 'csv']:
                    separator = '\t' if file_type == 'tsv' else ','
                    df = pd.read_csv(uploaded_result, sep=separator)
                    
                    st.success(f"✅ Loaded {len(df)} rows from {uploaded_result.name}")
                    
                    # Display with filters
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if 'table_name' in df.columns:
                            selected_tables = st.multiselect(
                                "Filter by Table",
                                options=df['table_name'].unique(),
                                default=df['table_name'].unique()[:5]
                            )
                            df = df[df['table_name'].isin(selected_tables)] if selected_tables else df
                    
                    with col2:
                        st.dataframe(df, use_container_width=True, height=400)
                    
                    # Download filtered data
                    if not df.empty:
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            "⬇️ Download Filtered Data",
                            data=csv_data,
                            file_name=f"filtered_{uploaded_result.name}",
                            mime="text/csv"
                        )
                
                elif file_type == 'sql':
                    content = uploaded_result.read().decode('utf-8')
                    st.success(f"✅ Loaded SQL file: {uploaded_result.name}")
                    
                    # Display SQL with syntax highlighting
                    st.code(content, language='sql')
                    
                    # Option to download
                    st.download_button(
                        "⬇️ Download SQL",
                        data=content,
                        file_name=uploaded_result.name,
                        mime="text/sql"
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
            help="Enter fully qualified table name to view metadata"
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
                            "tags": ["PII"]
                        },
                        {
                            "name": "column2", 
                            "type": "int",
                            "comment": "Another AI-generated description",
                            "tags": ["Non-PII"]
                        }
                    ]
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
            completed_jobs = sum(1 for job in st.session_state.job_runs.values() if job["status"] == "SUCCESS")
            failed_jobs = sum(1 for job in st.session_state.job_runs.values() if job["status"] == "FAILED")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Jobs", total_jobs)
            with col2:
                st.metric("Completed", completed_jobs)
            with col3:
                st.metric("Failed", failed_jobs)
            with col4:
                st.metric("Success Rate", f"{(completed_jobs/total_jobs*100):.1f}%" if total_jobs > 0 else "0%")
        else:
            st.info("ℹ️ No job statistics available yet")

    def run(self):
        """Main app execution"""
        # Header
        st.title("🏷️ DBX MetaGen")
        st.markdown("### AI-Powered Metadata Generation for Databricks Tables")
        
        # Sidebar configuration
        self.render_sidebar_config()
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["📋 Tables & Jobs", "📊 Results", "❓ Help"])
        
        with tab1:
            self.render_table_management()
            st.markdown("---")
            self.render_job_management()
        
        with tab2:
            self.render_results_viewer()
        
        with tab3:
            self.render_help()

    def render_help(self):
        """Render help and documentation"""
        st.header("❓ Help & Documentation")
        
        with st.expander("🚀 Getting Started", expanded=True):
            st.markdown("""
            1. **Configure Settings**: Use the sidebar to set your Databricks catalog, host, and processing options
            2. **Select Tables**: Either manually enter table names or upload a CSV file with table names
            3. **Create Job**: Configure and create a metadata generation job
            4. **Monitor Progress**: Watch job status and get notified when complete
            5. **View Results**: Browse generated metadata, download files, and review table comments
            """)
        
        with st.expander("⚙️ Configuration Options"):
            st.markdown("""
            - **Catalog Name**: Target catalog for storing metadata results
            - **Allow Data**: Whether to include actual data samples in LLM processing
            - **Sample Size**: Number of data rows to sample per column (0 = no data sampling)
            - **Mode**: Choose between generating comments, identifying PII, or both
            - **Apply DDL**: ⚠️ **WARNING** - This will directly modify your tables
            """)
        
        with st.expander("📁 File Formats"):
            st.markdown("""
            - **table_names.csv**: CSV file with 'table_name' column containing fully qualified table names
            - **Generated TSV**: Tab-separated file with metadata results
            - **Generated SQL**: DDL statements to apply metadata to tables
            """)
        
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            - **Authentication Issues**: Ensure Databricks token and host are properly configured
            - **Table Access**: Verify you have read access to the tables you want to process
            - **Job Failures**: Check job logs in Databricks workspace for detailed error information
            - **Large Tables**: Consider reducing sample size for very large tables
            """)

# Run the app
if __name__ == "__main__":
    app = DBXMetaGenApp()
    app.run()