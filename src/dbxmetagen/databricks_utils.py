"""Databricks environment setup utilities."""

import os
import json
from databricks.sdk import WorkspaceClient


def setup_databricks_environment(dbutils_instance=None):
    """Set up Databricks environment variables and return current user."""
    current_user = None

    # Try WorkspaceClient for user info and host
    try:
        w = WorkspaceClient()
        current_user = w.current_user.me().user_name

        if w.config.host:
            os.environ["DATABRICKS_HOST"] = w.config.host.rstrip("/")

        print(f"✓ Successfully authenticated as: {current_user}")

    except Exception:
        # Fallback to SQL for user (works in notebook environment)
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.getActiveSession()
            if spark:
                current_user = spark.sql("SELECT current_user()").collect()[0][0]

                # Try to get workspace URL from Spark config
                workspace_url = spark.conf.get("spark.databricks.workspaceUrl", None)
                if workspace_url:
                    if not workspace_url.startswith("https://"):
                        workspace_url = f"https://{workspace_url}"
                    os.environ["DATABRICKS_HOST"] = workspace_url
        except Exception:
            print("Warning: Could not get user info from Spark")

    # For API token, use the original dbutils approach
    try:
        if dbutils_instance:
            api_token = (
                dbutils_instance.notebook.entry_point.getDbutils()
                .notebook()
                .getContext()
                .apiToken()
                .get()
            )
            os.environ["DATABRICKS_TOKEN"] = api_token
        else:
            print("Warning: dbutils not provided - DATABRICKS_TOKEN not set")
    except Exception as e:
        print(f"Warning: Could not set DATABRICKS_TOKEN: {e}")

    return current_user


def get_job_context(dbutils_instance=None):
    """Get job context information if running in a job."""
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.getActiveSession()
        if spark:
            job_id = spark.conf.get("spark.databricks.clusterUsageTags.jobId", None)
            if job_id:
                return job_id

        if dbutils_instance:
            context_json = (
                dbutils_instance.notebook.entry_point.getDbutils()
                .notebook()
                .getContext()
                .toJson()
            )
            context = json.loads(context_json)
            return context.get("tags", {}).get("jobId", None)

        return None
    except Exception:
        return None
