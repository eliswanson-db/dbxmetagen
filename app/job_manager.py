import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from databricks.sdk.service.jobs import (
    Task,
    NotebookTask,
    JobEmailNotifications,
    JobCluster,
)
from databricks.sdk.service.compute import ClusterSpec, RuntimeEngine

# Set up logger for this module
logger = logging.getLogger(__name__)

# Import streamlit for debug logging (if/ available)
try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


def debug_log_job_manager(message: str):
    """Debug logging for job manager that works with or without Streamlit"""
    logger.debug(message)

    # If running in Streamlit and debug mode is on, add to session state
    if STREAMLIT_AVAILABLE and os.environ.get("DEBUG", "false").lower() == "true":
        try:
            if hasattr(st, "session_state") and "debug_logs" in st.session_state:
                timestamp = datetime.now().strftime("%H:%M:%S")
                debug_entry = f"[JOB_MGR] {timestamp} - {message}"
                st.session_state.debug_logs.append(debug_entry)

                # Keep only last 50 debug messages
                if len(st.session_state.debug_logs) > 50:
                    st.session_state.debug_logs = st.session_state.debug_logs[-50:]
        except Exception:
            # Ignore errors in debug logging
            pass


class DBXMetaGenJobManager:
    """Simplified job manager specifically for DBXMetaGen app"""

    def __init__(self, workspace_client):
        self.workspace_client = workspace_client

    def _resolve_notebook_path(self, config: Dict[str, Any]) -> str:
        """Determine the workspace notebook path to run based on env/config.

        Precedence:
        1) Explicit override via NOTEBOOK_PATH env or config['notebook_path']
        2) Use configured deploy user (config['owner_user'] or DEPLOY_USER_NAME env)
        3) If APP_ENV == 'dev' and a human user can be inferred from current_user, use that
        4) Fallback to Shared path
        """
        # 1) Explicit override
        explicit_path = os.environ.get("NOTEBOOK_PATH") or config.get("notebook_path")
        if explicit_path:
            debug_log_job_manager(
                f"Using explicit NOTEBOOK_PATH override: {explicit_path}"
            )
            return explicit_path

        # Determine bundle target and app env
        bundle_target = config.get("bundle_target") or os.environ.get(
            "BUNDLE_TARGET", "dev"
        )
        app_env = (os.environ.get("APP_ENV", "dev") or "dev").lower()

        # Helper: detect if a string looks like a service principal GUID (avoid using as user folder)
        def _looks_like_sp_guid(value: Optional[str]) -> bool:
            if not value:
                return False
            v = value.strip().lower()
            # basic GUID heuristic; also exclude emails (contain @)
            return (
                "@" not in v
                and len(v) == 36
                and all(c in "0123456789abcdef-" for c in v)
                and v.count("-") == 4
            )

        # 2) Prefer configured deploy user (owner_user variable or DEPLOY_USER_NAME env)
        configured_owner = (config.get("owner_user") or "").strip()
        if (
            configured_owner
            and "${" not in configured_owner
            and "@" in configured_owner
        ):
            path = f"/Workspace/Users/{configured_owner}/.bundle/dbxmetagen/{bundle_target}/files/notebooks/generate_metadata.py"
            debug_log_job_manager(f"Resolved notebook path from owner_user: {path}")
            return path

        deploy_user = os.environ.get("DEPLOY_USER_NAME", "").strip()
        if deploy_user and "@" in deploy_user:
            path = f"/Workspace/Users/{deploy_user}/.bundle/dbxmetagen/{bundle_target}/files/notebooks/generate_metadata.py"
            debug_log_job_manager(
                f"Resolved notebook path from DEPLOY_USER_NAME: {path}"
            )
            return path

        # 3) In dev, attempt to use current_user if it looks like a human email
        if app_env == "dev":
            try:
                current_user = self.workspace_client.current_user.me()
                candidate = (getattr(current_user, "user_name", "") or "").strip()
                if (
                    candidate
                    and not _looks_like_sp_guid(candidate)
                    and "@" in candidate
                ):
                    path = f"/Workspace/Users/{candidate}/.bundle/dbxmetagen/{bundle_target}/files/notebooks/generate_metadata"
                    debug_log_job_manager(
                        f"Resolved dev notebook path from current_user: {path}"
                    )
                    return path
                else:
                    debug_log_job_manager(
                        f"Current user does not look like a human email ('{candidate}'), avoiding SP home"
                    )
            except Exception as e:
                debug_log_job_manager(f"Failed to get current user for dev path: {e}")

        # 4) Fallback to Shared path (ensure files are deployed there if you use this)
        shared_base = os.environ.get("SHARED_WORKSPACE_BASE", "/Workspace/Shared")
        path = f"{shared_base}/dbxmetagen/{bundle_target}/files/notebooks/generate_metadata"
        debug_log_job_manager(f"Resolved fallback (Shared) notebook path: {path}")
        return path

    def create_metadata_job(
        self,
        job_name: str,
        tables: List[str],
        cluster_size: str,
        config: Dict[str, Any],
        user_email: Optional[str] = None,
    ) -> tuple[int, int]:
        """Create and run a metadata generation job"""

        print(f"[JOB_MGR] create_metadata_job called with job_name={job_name}")
        print(f"[JOB_MGR] Tables count: {len(tables)}")
        print(f"[JOB_MGR] Cluster size: {cluster_size}")
        print(f"[JOB_MGR] Config keys: {list(config.keys())}")

        logger.debug(f"Creating metadata job: {job_name}")
        logger.debug(f"Tables: {len(tables)} tables")
        logger.debug(f"Cluster size: {cluster_size}")
        logger.debug(f"Config keys: {list(config.keys())}")

        debug_log_job_manager(f"Creating metadata job: {job_name}")
        debug_log_job_manager(f"Tables: {len(tables)} tables")
        debug_log_job_manager(f"Cluster size: {cluster_size}")
        debug_log_job_manager(f"Config keys: {list(config.keys())}")

        try:
            print(f"[JOB_MGR] Starting job creation process...")
            # Map cluster size to worker count
            worker_map = {
                "Small (1-2 workers)": {"min": 1, "max": 2},
                "Medium (2-4 workers)": {"min": 2, "max": 4},
                "Large (4-8 workers)": {"min": 4, "max": 8},
            }

            if cluster_size not in worker_map:
                error_msg = f"Invalid cluster size: {cluster_size}. Must be one of {list(worker_map.keys())}"
                print(f"[JOB_MGR] ERROR: {error_msg}")
                raise ValueError(error_msg)

            workers = worker_map[cluster_size]
            debug_log_job_manager(f"Using worker config: {workers}")
            print(f"[JOB_MGR] Using worker config: {workers}")

            # Prepare job parameters
            print(f"[JOB_MGR] Preparing job parameters...")

            # Debug: check what mode is being set
            mode_value = config.get("mode", "comment")
            print(
                f"[JOB_MGR] Mode from config: {mode_value} (config keys: {list(config.keys())})"
            )

            job_parameters = {
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
                "reviewable_output_format": config.get(
                    "reviewable_output_format", "tsv"
                ),
                # Additional flags
                "disable_medical_information_value": str(
                    config.get("disable_medical_information_value", True)
                ).lower(),
                "add_metadata": str(config.get("add_metadata", True)).lower(),
                "include_deterministic_pi": str(
                    config.get("include_deterministic_pi", True)
                ).lower(),
            }

            debug_log_job_manager(
                f"Job parameters prepared: {len(job_parameters)} parameters"
            )
            print(
                f"[JOB_MGR] Job parameters prepared: {len(job_parameters)} parameters"
            )

            # Auto-detect cloud provider and node type
            workspace_url = config.get("host", "")
            if "azure" in workspace_url.lower():
                node_type = "Standard_D3_v2"
            elif "aws" in workspace_url.lower():
                node_type = "i3.xlarge"
            elif "gcp" in workspace_url.lower():
                node_type = "n1-standard-4"
            else:
                node_type = "Standard_D3_v2"  # Default to Azure

            debug_log_job_manager(
                f"Selected node type: {node_type} for workspace: {workspace_url}"
            )
            print(
                f"[JOB_MGR] Selected node type: {node_type} for workspace: {workspace_url}"
            )

            # Resolve notebook path now that we know env
            notebook_path = self._resolve_notebook_path(config)
            print(f"[JOB_MGR] Using notebook path: {notebook_path}")
            debug_log_job_manager(f"Using notebook path: {notebook_path}")

            # Best-effort existence check for path visibility
            try:
                status = self.workspace_client.workspace.get_status(notebook_path)
                debug_log_job_manager(
                    f"Notebook path status: {getattr(status, 'object_type', 'unknown')}"
                )
            except Exception as status_e:
                debug_log_job_manager(f"Could not verify notebook path: {status_e}")

            # Create job using the official SDK pattern (matching documentation examples)
            print(f"[JOB_MGR] Creating job in Databricks...")
            try:
                job = self.workspace_client.jobs.create(
                    name=job_name,
                    job_clusters=[
                        JobCluster(
                            job_cluster_key="metadata_cluster",
                            new_cluster=ClusterSpec(
                                spark_version="14.3.x-cpu-ml-scala2.12",
                                node_type_id=node_type,
                                num_workers=workers["min"],
                                runtime_engine=RuntimeEngine.STANDARD,
                            ),
                        )
                    ],
                    tasks=[
                        Task(
                            description="Generate metadata for tables",
                            task_key="generate_metadata",
                            notebook_task=NotebookTask(
                                notebook_path=notebook_path,
                                base_parameters=job_parameters,
                            ),
                            job_cluster_key="metadata_cluster",
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
                logger.info(f"Job created successfully with ID: {job.job_id}")
                print(f"[JOB_MGR] Job created successfully with ID: {job.job_id}")
            except Exception as create_e:
                error_msg = f"Job creation failed: {str(create_e)}"
                print(f"[JOB_MGR] ERROR creating job: {error_msg}")
                logger.error(f"Failed to create job: {str(create_e)}")
                raise RuntimeError(error_msg)

            # Job permissions would be set here if needed
            if user_email:
                logger.info(
                    f"Job permissions for {user_email} should be set manually via Databricks UI if needed"
                )
                print(
                    f"[JOB_MGR] Note: Permissions for {user_email} should be set manually via Databricks UI if needed"
                )

            # Run the job
            logger.debug(f"Starting job run for job ID: {job.job_id}")
            print(f"[JOB_MGR] Starting job run for job ID: {job.job_id}")
            try:
                run = self.workspace_client.jobs.run_now(
                    job_id=job.job_id,
                    notebook_params=job_parameters,
                )
                logger.info(f"Job run started successfully with run ID: {run.run_id}")
                print(
                    f"[JOB_MGR] Job run started successfully with run ID: {run.run_id}"
                )
            except Exception as run_e:
                error_msg = f"Job run failed to start: {str(run_e)}"
                print(f"[JOB_MGR] ERROR starting job run: {error_msg}")
                logger.error(f"Failed to start job run: {str(run_e)}")
                raise RuntimeError(error_msg)

            print(f"[JOB_MGR] Returning job_id={job.job_id}, run_id={run.run_id}")
            return job.job_id, run.run_id

        except Exception as e:
            print(f"[JOB_MGR] EXCEPTION in create_metadata_job: {str(e)}")
            print(f"[JOB_MGR] Exception type: {type(e).__name__}")
            logger.error(f"create_metadata_job failed: {str(e)}", exc_info=True)
            raise

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
