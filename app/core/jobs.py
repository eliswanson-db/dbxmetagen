"""
Job management module.
Handles Databricks job creation, monitoring, and status tracking.
"""

import streamlit as st
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime
from databricks.sdk.service.jobs import (
    JobSettings,
    NotebookTask,
    JobCluster,
    ClusterSpec,
    JobEnvironment,
    Task,
    PerformanceTarget,
)
from databricks.sdk.service.compute import Environment

# Import job manager with error handling
try:
    from job_manager import DBXMetaGenJobManager

    JOB_MANAGER_AVAILABLE = True
    JOB_MANAGER_IMPORT_ERROR = None
except ImportError as e:
    JOB_MANAGER_AVAILABLE = False
    JOB_MANAGER_IMPORT_ERROR = str(e)

logger = logging.getLogger(__name__)


class JobManager:
    """Manages Databricks job operations for metadata generation."""

    def __init__(self):
        self.job_manager_available = JOB_MANAGER_AVAILABLE
        self.job_manager_import_error = JOB_MANAGER_IMPORT_ERROR

    def create_and_run_job(self, job_name: str, tables: List[str], cluster_size: str):
        """Create and run a metadata generation job."""
        if not self._validate_job_prerequisites(tables):
            return

        try:
            # Load fresh configuration
            config = self._load_fresh_config_from_yaml()

            # Extract user information
            username, user_email = self._extract_user_information()

            # Fix any config contamination
            config = self._fix_config_user_contamination(config, username)

            # Initialize job manager
            job_manager = self._initialize_job_manager()

            # Generate unique job name
            unique_job_name = self._generate_unique_job_name(username)

            # Clean up existing jobs
            self._cleanup_existing_jobs(unique_job_name)

            # Format tables for processing
            tables_string = self._format_tables_for_processing(tables)

            # Create and start the job
            job_id, run_id = self._create_and_start_job(
                job_manager, unique_job_name, tables_string, cluster_size, config
            )

            # Store job information and show success
            self._store_job_info_and_show_success(
                job_id, run_id, unique_job_name, tables, cluster_size
            )

        except Exception as e:
            logger.error(f"Failed to create job: {str(e)}")
            st.error(f"❌ Failed to create job: {str(e)}")

    def _validate_job_prerequisites(self, tables: List[str]) -> bool:
        """Validate that job can be created."""
        if not self.job_manager_available:
            st.error(f"❌ Job Manager not available: {self.job_manager_import_error}")
            return False

        if not st.session_state.get("workspace_client"):
            st.error("❌ Workspace client not initialized")
            return False

        if not tables:
            st.error("❌ No tables provided")
            return False

        return True

    def _load_fresh_config_from_yaml(self) -> Dict[str, Any]:
        """Load fresh configuration from YAML file."""
        from core.config import ConfigManager

        config_manager = ConfigManager()
        return config_manager.load_default_config()

    def _extract_user_information(self) -> Tuple[str, str]:
        """Extract current user information."""
        try:
            current_user = st.session_state.workspace_client.current_user.me()
            return current_user.user_name, current_user.user_name
        except Exception as e:
            logger.warning(f"Could not get current user: {str(e)}")
            return "unknown_user", "unknown@example.com"

    def _fix_config_user_contamination(
        self, config: Dict[str, Any], username: str
    ) -> Dict[str, Any]:
        """Remove any user-specific contamination from config."""
        config = config.copy()

        # Remove user-specific paths that might cause issues
        user_contamination_keys = [
            "current_user_name",
            "user_home",
            "workspace_path",
            "user_email",
            "personal_folder",
        ]

        for key in user_contamination_keys:
            if key in config:
                logger.debug(f"Removed user contamination key: {key}")
                del config[key]

        # Ensure we have clean, generic values
        config.setdefault("log_timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))

        return config

    def _initialize_job_manager(self):
        """Initialize the job manager."""
        return DBXMetaGenJobManager(st.session_state.workspace_client)

    def _generate_unique_job_name(self, username: str) -> str:
        """Generate a unique job name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Clean username for job naming
        clean_username = username.replace("@", "_").replace(".", "_")
        return f"dbxmetagen_{clean_username}_{timestamp}"

    def _cleanup_existing_jobs(self, job_name: str):
        """Clean up any existing jobs with similar names."""
        try:
            existing_jobs = st.session_state.workspace_client.jobs.list(
                name=job_name, limit=10
            )

            for job in existing_jobs:
                if job.settings and job.settings.name == job_name:
                    logger.info(f"Found existing job {job.job_id}, will be replaced")

        except Exception as e:
            logger.warning(f"Could not check for existing jobs: {str(e)}")

    def _format_tables_for_processing(self, tables: List[str]) -> str:
        """Format table list for job processing."""
        # Join tables with newlines for processing
        return "\n".join(tables)

    def _create_and_start_job(
        self,
        job_manager,
        job_name: str,
        tables_string: str,
        cluster_size: str,
        config: Dict[str, Any],
    ) -> Tuple[int, int]:
        """Create and start the actual job."""
        # Create metadata generation job
        job_id, run_id = job_manager.create_metadata_job(
            job_name=job_name,
            table_names=tables_string,
            cluster_size=cluster_size,
            config=config,
        )

        logger.info(f"Job created successfully - ID: {job_id}, Run: {run_id}")
        return job_id, run_id

    def _store_job_info_and_show_success(
        self,
        job_id: int,
        run_id: int,
        job_name: str,
        tables: List[str],
        cluster_size: str,
    ):
        """Store job information in session state and show success message."""
        # Store job information
        if "job_runs" not in st.session_state:
            st.session_state.job_runs = {}

        st.session_state.job_runs[job_id] = {
            "run_id": run_id,
            "job_name": job_name,
            "tables": tables,
            "cluster_size": cluster_size,
            "created_at": datetime.now().isoformat(),
            "status": "RUNNING",
        }

        # Show success message
        st.success(
            f"✅ Job '{job_name}' created and started!\n"
            f"📋 Job ID: {job_id}\n"
            f"🔄 Run ID: {run_id}\n"
            f"📊 Processing {len(tables)} tables"
        )

        # Show job URL if possible
        try:
            host = st.session_state.workspace_client.config.host
            job_url = f"{host}/#job/{job_id}/run/{run_id}"
            st.info(f"🔗 [View Job in Databricks]({job_url})")
        except Exception:
            pass

    def refresh_job_status(self):
        """Refresh status of all tracked jobs."""
        if not st.session_state.get("job_runs"):
            st.info("No jobs to track")
            return

        if not st.session_state.get("workspace_client"):
            st.error("❌ Workspace client not initialized")
            return

        try:
            updated_jobs = {}

            for job_id, job_info in st.session_state.job_runs.items():
                try:
                    # Get current run status
                    run_id = job_info["run_id"]
                    run_status = st.session_state.workspace_client.jobs.get_run(run_id)

                    # Update job info
                    job_info["status"] = run_status.state.life_cycle_state.value
                    job_info["result_state"] = (
                        run_status.state.result_state.value
                        if run_status.state.result_state
                        else None
                    )
                    job_info["last_updated"] = datetime.now().isoformat()

                    updated_jobs[job_id] = job_info

                    logger.info(f"Job {job_id} status: {job_info['status']}")

                except Exception as e:
                    logger.error(f"Failed to get status for job {job_id}: {str(e)}")
                    job_info["status"] = f"ERROR: {str(e)}"
                    updated_jobs[job_id] = job_info

            # Update session state
            st.session_state.job_runs = updated_jobs
            st.success(f"✅ Refreshed status for {len(updated_jobs)} jobs")

        except Exception as e:
            logger.error(f"Failed to refresh job status: {str(e)}")
            st.error(f"❌ Failed to refresh job status: {str(e)}")

    def create_sync_metadata_job(self, df, filename: str):
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

            # Create job configuration
            job_config = JobSettings(
                environments=[
                    JobEnvironment(
                        environment_key="default_python",
                        spec=Environment(environment_version="1"),
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
                    )
                ],
            )

            # Create the job
            job = st.session_state.workspace_client.jobs.create(**job_config.as_dict())

            st.success(f"✅ Sync job created! Job ID: {job.job_id}")
            st.info(
                "📋 Upload your reviewed metadata file to the volume and then run this job."
            )

        except Exception as e:
            logger.error(f"Error creating sync job: {str(e)}")
            st.error(f"❌ Error creating sync job: {str(e)}")
