import os
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from databricks.sdk.service.jobs import (
    NotebookTask,
    JobEnvironment,
    Task,
    PerformanceTarget,
    JobEmailNotifications,
)
from databricks.sdk.service.compute import Environment

logger = logging.getLogger(__name__)


class DBXJobManager:
    """Handles Databricks job creation and monitoring - clean and simple"""

    def __init__(self, workspace_client: WorkspaceClient):
        self.workspace_client = workspace_client

    def create_metadata_job(
        self,
        job_name: str,
        tables: List[str],
        config: Dict[str, Any],
        user_email: Optional[str] = None,
    ) -> Tuple[int, int]:
        """Create and run a metadata generation job"""
        logger.info(f"Creating metadata job: {job_name}")
        w = WorkspaceClient()
        notebook_path = f"/Users/{w.current_user.me().user_name}/"

        if config.get("cluster_id"):
            created_job = w.jobs.create(
                name=f"sdk-{time.time_ns()}",
                tasks=[
                    jobs.Task(
                        description="test",
                        existing_cluster_id=config.get("cluster_id"),
                        notebook_task=NotebookTask(notebook_path=notebook_path),
                        task_key="test",
                        timeout_seconds=0,
                    )
                ],
            )
        else:
            created_job = w.jobs.create(
                name=f"sdk-{time.time_ns()}",
                tasks=[
                    jobs.Task(
                        description="test",
                        notebook_task=jobs.NotebookTask(notebook_path=notebook_path),
                        task_key="test",
                        timeout_seconds=0,
                    )
                ],
            )

        return created_job.job_id, created_job.run_id


class JobManager:
    """Handles Databricks job creation and monitoring - clean and simple"""

    def __init__(self, workspace_client: WorkspaceClient):
        self.workspace_client = workspace_client

    def create_metadata_job(
        self,
        job_name: str,
        tables: List[str],
        cluster_size: str,
        config: Dict[str, Any],
        user_email: Optional[str] = None,
    ) -> Tuple[int, int]:
        """Create and run a metadata generation job"""
        logger.info(f"Creating metadata job: {job_name}")

        # Step 1: Validate inputs
        self._validate_inputs(job_name, tables, cluster_size, config)

        # Step 2: Prepare job configuration
        workers = self._get_worker_config(cluster_size)
        job_parameters = self._build_job_parameters(tables, config)
        node_type = self._detect_node_type(config)
        notebook_path = self._resolve_notebook_path(config)

        # Step 3: Create and run job
        job_id = self._create_job(job_name, notebook_path, job_parameters, user_email)
        run_id = self._start_job_run(job_id, job_parameters)

        logger.info(f"Job created successfully - job_id: {job_id}, run_id: {run_id}")
        return job_id, run_id

    def get_run_status(self, run_id: int) -> Dict[str, Any]:
        """Get the status of a job run"""
        try:
            run_details = self.workspace_client.jobs.get_run(run_id)
            return {
                "status": run_details.state.life_cycle_state.value,
                "result_state": (
                    run_details.state.result_state.value
                    if run_details.state.result_state
                    else None
                ),
                "start_time": run_details.start_time,
                "end_time": run_details.end_time,
                "run_page_url": run_details.run_page_url,
            }
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def refresh_job_status(self):
        """Refresh status of all tracked jobs"""
        if not st.session_state.get("job_runs"):
            st.info("No jobs to track")
            return

        try:
            for run_id, job_info in st.session_state.job_runs.items():
                try:
                    run_status = self.workspace_client.jobs.get_run(int(run_id))
                    old_status = job_info.get("status", "UNKNOWN")
                    new_status = run_status.state.life_cycle_state.value

                    job_info["status"] = new_status
                    job_info["last_updated"] = datetime.now().isoformat()

                    if run_status.state.result_state:
                        job_info["result_state"] = run_status.state.result_state.value

                    if old_status != new_status:
                        logger.info(
                            f"Run {run_id} status changed: {old_status} -> {new_status}"
                        )

                except Exception as e:
                    logger.error(f"Failed to get status for run {run_id}: {str(e)}")
                    job_info["status"] = f"ERROR: {str(e)}"

            st.success(
                f"✅ Refreshed status for {len(st.session_state.job_runs)} job runs"
            )

        except Exception as e:
            logger.error(f"Failed to refresh job status: {str(e)}")
            st.error(f"❌ Failed to refresh job status: {str(e)}")

    # === Private helper methods (small and composable) ===

    def _validate_inputs(
        self,
        job_name: str,
        tables: List[str],
        cluster_size: str,
        config: Dict[str, Any],
    ):
        """Validate all inputs for job creation"""
        if not job_name:
            raise ValueError("Job name is required")
        if not tables:
            raise ValueError("No tables specified for processing")
        if not config:
            raise ValueError("Configuration is required")

        valid_sizes = [
            "Small (1-2 workers)",
            "Medium (2-4 workers)",
            "Large (4-8 workers)",
        ]
        if cluster_size not in valid_sizes:
            raise ValueError(
                f"Invalid cluster size: {cluster_size}. Must be one of {valid_sizes}"
            )

    def _get_worker_config(self, cluster_size: str) -> Dict[str, int]:
        """Map cluster size to worker configuration"""
        worker_map = {
            "Small (1-2 workers)": {"min": 1, "max": 2},
            "Medium (2-4 workers)": {"min": 2, "max": 4},
            "Large (4-8 workers)": {"min": 4, "max": 8},
        }
        return worker_map[cluster_size]

    def _build_job_parameters(
        self, tables: List[str], config: Dict[str, Any]
    ) -> Dict[str, str]:
        """Build job parameters for notebook execution"""
        mode_value = config.get("mode", "comment")

        return {
            "table_names": tables if isinstance(tables, str) else ",".join(tables),
            "env": "app",
            "cleanup_control_table": "true",
            # Core settings
            "catalog_name": config.get("catalog_name", "dbxmetagen"),
            "host": config.get("host", ""),
            "schema_name": config.get("schema_name", "metadata_results"),
            "volume_name": config.get("volume_name", "generated_metadata"),
            # Data settings
            "allow_data": str(config.get("allow_data", False)).lower(),
            "sample_size": str(config.get("sample_size", 5)),
            "mode": mode_value,
            "allow_data_in_comments": str(
                config.get("allow_data_in_comments", True)
            ).lower(),
            # Advanced settings
            "model": config.get("model", "databricks-claude-3-7-sonnet"),
            "max_tokens": str(config.get("max_tokens", 4096)),
            "temperature": str(config.get("temperature", 0.1)),
            "columns_per_call": str(config.get("columns_per_call", 5)),
            "apply_ddl": str(config.get("apply_ddl", False)).lower(),
            "ddl_output_format": config.get("ddl_output_format", "sql"),
            "reviewable_output_format": config.get("reviewable_output_format", "tsv"),
            # Additional flags
            "disable_medical_information_value": str(
                config.get("disable_medical_information_value", True)
            ).lower(),
            "add_metadata": str(config.get("add_metadata", True)).lower(),
            "include_deterministic_pi": str(
                config.get("include_deterministic_pi", True)
            ).lower(),
        }

    def _detect_node_type(self, config: Dict[str, Any]) -> str:
        """Auto-detect appropriate node type based on workspace"""
        workspace_url = config.get("host", "")
        if "azure" in workspace_url.lower():
            return "Standard_D3_v2"
        elif "aws" in workspace_url.lower():
            return "i3.xlarge"
        elif "gcp" in workspace_url.lower():
            return "n1-standard-4"
        else:
            return "Standard_D3_v2"  # Default to Azure

    def _resolve_notebook_path(self, config: Dict[str, Any]) -> str:
        """Resolve the notebook path for job execution"""
        # 1) Explicit override
        explicit_path = os.environ.get("NOTEBOOK_PATH") or config.get("notebook_path")
        if explicit_path:
            logger.info(f"Using explicit NOTEBOOK_PATH override: {explicit_path}")
            return explicit_path

        # 2) Try current workspace user
        try:
            current_user = self.workspace_client.current_user.me()
            user_name = current_user.user_name

            # Check if this looks like a real user email (not service principal GUID)
            if "@" in user_name and not user_name.startswith("034f50f1"):
                bundle_target = config.get("bundle_target", "dev")
                path = f"/Workspace/Users/{user_name}/.bundle/dbxmetagen/{bundle_target}/files/notebooks/generate_metadata"
                logger.info(f"Resolved notebook path for user {user_name}: {path}")
                return path
            else:
                logger.info(
                    f"Current user appears to be service principal: {user_name}"
                )

        except Exception as e:
            logger.warning(f"Could not resolve current user path: {e}")

        # 3) Fallback to hardcoded path (TODO: make this dynamic)
        bundle_target = config.get("bundle_target", "dev")
        path = f"/Workspace/Users/eli.swanson@databricks.com/.bundle/dbxmetagen/{bundle_target}/files/notebooks/generate_metadata"
        logger.info(f"Using fallback path: {path}")
        return path

    def _find_existing_cluster(self) -> Optional[str]:
        """Find an existing cluster that can be reused"""
        try:
            clusters = self.workspace_client.clusters.list()
            for cluster in clusters:
                if (
                    cluster.state
                    and cluster.state.value in ["RUNNING", "TERMINATED"]
                    and hasattr(cluster, "cluster_name")
                    and "shared" in cluster.cluster_name.lower()
                ):
                    logger.info(
                        f"Found existing cluster: {cluster.cluster_name} ({cluster.cluster_id})"
                    )
                    return cluster.cluster_id
        except Exception as e:
            logger.warning(f"Could not list clusters: {e}")
        return None

    def _create_job(
        self,
        job_name: str,
        notebook_path: str,
        job_parameters: Dict[str, str],
        user_email: Optional[str] = None,
    ) -> int:
        """Create the Databricks job"""
        # Validate critical parameters
        if not notebook_path:
            raise ValueError("Notebook path is empty - cannot create job")
        if not job_parameters.get("table_names"):
            raise ValueError("No table names provided - cannot create job")

        existing_cluster_id = self._find_existing_cluster()

        try:
            job = self.workspace_client.jobs.create(
                environments=[
                    JobEnvironment(
                        environment_key="default_python",
                        spec=Environment(environment_version="2"),
                    )
                ],
                performance_target=PerformanceTarget.PERFORMANCE_OPTIMIZED,
                name=job_name,
                tasks=[
                    Task(
                        description="Generate metadata for tables",
                        task_key="generate_metadata",
                        notebook_task=NotebookTask(
                            notebook_path=notebook_path,
                            base_parameters=job_parameters,
                        ),
                        existing_cluster_id=existing_cluster_id,
                        timeout_seconds=14400,
                    )
                ],
                email_notifications=(
                    JobEmailNotifications(
                        on_failure=[user_email] if user_email else [],
                        on_success=[user_email] if user_email else [],
                    )
                    if user_email
                    else None
                ),
                max_concurrent_runs=1,
            )

            # Verify job creation
            created_job = self.workspace_client.jobs.get(job.job_id)
            task_count = (
                len(created_job.settings.tasks) if created_job.settings.tasks else 0
            )
            logger.info(f"Job created successfully with {task_count} tasks")

            return job.job_id

        except Exception as e:
            error_msg = f"Job creation failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _start_job_run(self, job_id: int, job_parameters: Dict[str, str]) -> int:
        """Start a job run and return the run ID"""
        try:
            run = self.workspace_client.jobs.run_now(
                job_id=job_id,
                notebook_params=job_parameters,
            )
            logger.info(f"Job run started successfully with run ID: {run.run_id}")
            return run.run_id

        except Exception as e:
            error_msg = f"Job run failed to start: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    # === Job creation UI methods ===

    def create_and_run_job(self, job_name: str, tables: List[str], cluster_size: str):
        """Create and run a metadata generation job (Streamlit UI wrapper)"""
        if not st.session_state.get("workspace_client"):
            st.error("❌ Workspace client not initialized")
            return

        if not tables:
            st.error("❌ No tables provided")
            return

        try:
            job_id, run_id = self.create_metadata_job(
                job_name=job_name,
                tables=tables,
                cluster_size=cluster_size,
                config=st.session_state.config,
            )

            # Store job run info for tracking (using run_id as key)
            if "job_runs" not in st.session_state:
                st.session_state.job_runs = {}

            st.session_state.job_runs[run_id] = {
                "job_id": job_id,
                "job_name": job_name,
                "run_id": run_id,
                "tables": tables,
                "config": st.session_state.config,
                "cluster_size": cluster_size,
                "status": "RUNNING",
                "start_time": datetime.now(),
                "created_at": datetime.now().isoformat(),
            }

            st.success(f"✅ Job '{job_name}' created and started!")
            st.write(f"📋 **Job ID:** {job_id}")
            st.write(f"🔄 **Run ID:** {run_id}")
            st.write(f"📊 **Processing:** {len(tables)} tables")

        except Exception as e:
            st.error(f"❌ Failed to create job: {str(e)}")
            logger.error(f"Failed to create job: {str(e)}", exc_info=True)

            # Show debug details (matching debug method behavior)
            st.markdown("**Debug Information:**")
            st.write(f"- Job name: {job_name}")
            st.write(f"- Tables: {len(tables)} total")
            st.write(f"- Cluster size: {cluster_size}")
            st.write(
                f"- Config available: {'Yes' if st.session_state.config else 'No'}"
            )
            st.write(
                f"- Workspace client: {'Yes' if st.session_state.get('workspace_client') else 'No'}"
            )

            # Show full exception details
            import traceback

            st.code(traceback.format_exc())

    def debug_job_manager_creation(self, tables: List[str]):
        """Debug job creation with a test job"""
        try:
            from utils import get_current_user_email

            current_user_email = get_current_user_email()
            job_name = f"dbxmetagen_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            job_id, run_id = self.create_metadata_job(
                job_name=job_name,
                tables=tables,
                cluster_size="Small (1-2 workers)",
                config=st.session_state.config,
            )

            # Store job run info for tracking (using run_id as key)
            if "job_runs" not in st.session_state:
                st.session_state.job_runs = {}

            st.session_state.job_runs[run_id] = {
                "job_id": job_id,
                "job_name": job_name,
                "run_id": run_id,
                "tables": tables,
                "config": st.session_state.config,
                "cluster_size": "Small (1-2 workers)",
                "status": "RUNNING",
                "start_time": datetime.now(),
                "created_at": datetime.now().isoformat(),
            }

            st.success("✅ Debug job created!")
            st.write(f"Job ID: {job_id}, Run ID: {run_id}")

        except Exception as e:
            st.error(f"❌ Debug job creation failed: {str(e)}")
            logger.error(f"Debug job creation failed: {str(e)}", exc_info=True)

    def create_sync_metadata_job(self, df, filename: str):
        """Create a job to sync reviewed metadata"""
        try:
            from pandas import DataFrame

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

            job = self.workspace_client.jobs.create(
                environments=[
                    JobEnvironment(
                        environment_key="default_python",
                        spec=Environment(environment_version="2"),
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
                        timeout_seconds=14400,
                    )
                ],
                max_concurrent_runs=1,
            )

            st.success(f"✅ Sync job created! Job ID: {job.job_id}")
            st.info(
                "📋 Upload your reviewed metadata file to the volume and then run this job."
            )

        except Exception as e:
            st.error(f"❌ Error creating sync job: {str(e)}")
            logger.error(f"Error creating sync job: {str(e)}")
